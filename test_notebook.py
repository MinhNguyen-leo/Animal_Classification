"""Offline smoke check: python test_notebook.py (no dataset/weight downloads)."""

import ast
import csv
import json
import linecache
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")


def main():
    notebook = json.loads(Path(__file__).with_name("animals10.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        filename = f"cell_{index}"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        tree = ast.parse(source, filename=filename)
        # Load imports, configuration and definitions; skip downloads/training/demo cells.
        if "import tensorflow as tf" in source or all(
            isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body
        ):
            exec(compile(tree, filename, "exec"), namespace)

    np, tf, keras = (namespace[name] for name in ("np", "tf", "keras"))
    Image, classes = namespace["Image"], namespace["CLASSES"]
    namespace["seed_everything"](42)
    pixel = tf.constant([[[[10., 20., 30.]]]])
    np.testing.assert_allclose(namespace["preprocess_images"](pixel, "custom"), pixel / 255)
    np.testing.assert_allclose(namespace["preprocess_images"](pixel, "resnet50"),
                               [[[[30 - 103.939, 20 - 116.779, 10 - 123.68]]]], atol=1e-5)

    with tempfile.TemporaryDirectory(prefix="animals10_tf_") as directory:
        root = Path(directory)
        data, prepared = root / "raw-img", root / "data"
        for index, label in enumerate(classes):
            folder = data / label
            folder.mkdir(parents=True)
            for i in range(10 + index):
                pixels = np.random.default_rng(index * 100 + i).integers(0, 256, (40, 48, 3), dtype=np.uint8)
                suffix = ".webp" if i == 0 else ".png"
                Image.fromarray(pixels).save(folder / f"{i}{suffix}")
        (data / classes[0] / "broken.jpg").write_bytes(b"invalid image")
        namespace["prepare"](SimpleNamespace(data=data, output=prepared, seed=42))
        rows = namespace["read_splits"](prepared / "splits.csv")
        assert len(rows) == sum(10 + i for i in range(10))
        assert "broken.jpg" in (prepared / "invalid_images.txt").read_text(encoding="utf-8")
        groups = [{r["path"] for r in rows if r["split"] == split} for split in ("train", "val", "test")]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

        # One image per class per split keeps real Keras training small on CPU.
        tiny_rows = [next(r for r in rows if r["split"] == split and r["label"] == label)
                     for split in ("train", "val", "test") for label in classes]
        splits = root / "tiny_splits.csv"
        with splits.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tiny_rows[0].keys())
            writer.writeheader()
            writer.writerows(tiny_rows)
        weights = tf.constant([1 + i / 10 for i in range(10)], tf.float32)
        images, labels, sample_weights = next(iter(namespace["loader"](data, tiny_rows, "val", 10, "custom", weights)))
        assert images.shape == (10, 224, 224, 3)
        assert float(tf.reduce_min(images)) >= 0 and float(tf.reduce_max(images)) <= 1
        np.testing.assert_allclose(sample_weights, tf.gather(weights, labels))
        image_path = data / tiny_rows[-1]["path"]
        with Image.open(image_path) as source:
            image = source.copy()
        # Validation/test and demo must produce identical pixels.
        test_images, _ = next(iter(namespace["loader"](data, tiny_rows, "test", 10, "custom")))
        np.testing.assert_allclose(test_images[-1], namespace["preprocess_images"](namespace["image_array"](image), "custom"))

        original_make_model = namespace["make_model"]
        namespace["make_model"] = lambda name, pretrained=False: original_make_model(name, pretrained=False)
        for architecture in ("custom", "resnet50"):
            keras.backend.clear_session()
            output = root / architecture
            namespace["train"](SimpleNamespace(
                data=data, splits=splits, output=output, model=architecture,
                seed=42, batch_size=10, class_weights=True, lr=1e-3,
                finetune_lr=1e-4, epochs=1, head_epochs=1, finetune_epochs=1,
            ))
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))
            assert len(history) == (1 if architecture == "custom" else 2)
            model, restored_architecture = namespace["load_checkpoint"](output / "best.keras")
            assert restored_architecture == architecture
            val = namespace["loader"](data, tiny_rows, "val", 10, architecture)
            x, y = next(iter(val))
            losses = keras.losses.sparse_categorical_crossentropy(y, model(x, training=False), from_logits=True)
            np.testing.assert_allclose(float(tf.reduce_mean(losses)), min(r["val_loss"] for r in history), rtol=1e-4)
            probabilities = namespace["predict_probabilities"](model, architecture, image)
            assert probabilities.shape == (10,) and np.isfinite(probabilities).all()
            np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-6)
            heat = namespace["gradcam"](model, architecture, image)
            assert heat.shape == (224, 224) and np.isfinite(heat).all()
            assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
            if architecture == "resnet50":
                namespace["configure_resnet"](model, "head")
                assert [layer.name for layer in model.layers if layer.trainable_weights] == ["classifier"]
                namespace["configure_resnet"](model, "finetune")
                assert model.get_layer("conv5_block1_1_conv").trainable
                assert not model.get_layer("conv4_block1_1_conv").trainable
                assert all(not layer.trainable for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization))
            namespace["evaluate"](SimpleNamespace(
                data=data, splits=splits, checkpoint=output / "best.keras", output=output / "test", batch_size=10,
            ))
            report = json.loads((output / "test" / "metrics.json").read_text(encoding="utf-8"))
            assert 0 <= report["accuracy"] <= 1
            assert (output / "test" / "confusion_matrix.png").is_file()
            # Check metadata guards and Gradio's real prediction function without launching a server.
            demo_namespace = dict(namespace, demo_model=model, demo_architecture=architecture)
            import gradio as gr
            demo_namespace["gr"] = gr
            demo_cell = next(c for c in notebook["cells"] if c["cell_type"] == "code" and "def predict(image):" in "".join(c["source"]))
            predict_node = next(n for n in ast.parse("".join(demo_cell["source"])).body if isinstance(n, ast.FunctionDef))
            exec(compile(ast.Module(body=[predict_node], type_ignores=[]), "gradio_predict", "exec"), demo_namespace)
            scores = demo_namespace["predict"](image)
            assert set(scores) == set(namespace["LABELS"].values())
            try:
                demo_namespace["predict"](None)
            except gr.Error:
                pass
            else:
                raise AssertionError("Missing image must be rejected")
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["classes"] = list(reversed(classes))
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            try:
                namespace["load_checkpoint"](output / "best.keras")
            except ValueError:
                pass
            else:
                raise AssertionError("Class order mismatch must be rejected")
            del model
            print(f"PASS: {architecture} training, checkpoint, inference, evaluation, Grad-CAM and Gradio")
    print("PASS: TensorFlow notebook smoke check")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="animals10_mpl_") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        main()
