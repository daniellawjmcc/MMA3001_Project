
"""Tests for the MMA3001 training and dataset-preparation scripts.

These tests inspect source code without running model training,
downloading pretrained weights or modifying source datasets.
"""

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATHS = {
    "faster_rcnn": PROJECT_ROOT / "training" / "train_faster_rcnn.py",
    "retinanet": PROJECT_ROOT / "training" / "train_retinanet.py",
    "yolo_baseline": PROJECT_ROOT / "training" / "train_yolo_baseline.py",
    "yolo_4class": PROJECT_ROOT / "training" / "train_yolo_4class.py",
    "prepare_dataset": (
        PROJECT_ROOT / "src" / "mma3001" / "prepare_dataset.py"
    ),
}


def parse_script(script_name: str) -> ast.Module:
    """Parse a project script without importing or executing it.

    Args:
        script_name: Key identifying the script to inspect.

    Returns:
        Parsed Python abstract syntax tree.

    Raises:
        KeyError: If the script name is not recognised.
        FileNotFoundError: If the source file is missing.
        SyntaxError: If the source contains invalid Python syntax.
    """
    path = SCRIPT_PATHS[script_name]

    if not path.is_file():
        raise FileNotFoundError(
            f"Required script not found: {path}"
        )

    return ast.parse(
        path.read_text(encoding="utf-8-sig"),
        filename=str(path),
    )


def get_source(script_name: str) -> str:
    """Read the original text of a reproduction script.

    Args:
        script_name: Key identifying the script.

    Returns:
        Complete Python source code as a string.
    """
    return SCRIPT_PATHS[script_name].read_text(
        encoding="utf-8-sig"
    )


def get_literal_constants(tree: ast.Module) -> dict:
    """Extract literal top-level configuration constants.

    Args:
        tree: Parsed Python module.

    Returns:
        Dictionary mapping constant names to literal values.
        Nonliteral expressions are excluded.
    """
    constants = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value

        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value

        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue

            try:
                constants[target.id] = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError):
                continue

    return constants


@pytest.mark.parametrize("script_name", SCRIPT_PATHS)
def test_scripts_parse_successfully(script_name: str) -> None:
    """Confirm each new project script has valid Python syntax.

    Args:
        script_name: Key identifying a reproduction script.

    Returns:
        None. Assertions verify successful parsing.
    """
    tree = parse_script(script_name)

    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("script_name", SCRIPT_PATHS)
def test_functions_have_google_style_docstrings(
    script_name: str,
) -> None:
    """Check Google-style documentation in the reproduction scripts.

    Args:
        script_name: Key identifying the source script.

    Returns:
        None. Assertions verify function docstrings.
    """
    tree = parse_script(script_name)

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert functions

    for function in functions:
        docstring = ast.get_docstring(function)

        assert docstring is not None, (
            f"{script_name}.{function.name} has no docstring."
        )

        assert "Returns:" in docstring, (
            f"{script_name}.{function.name} has no Returns section."
        )

        if function.args.args:
            assert "Args:" in docstring or not function.args.args, (
                f"{script_name}.{function.name} has no Args section."
            )

        # Raises documentation is checked where explicitly supplied.
        if "Raises:" in docstring:
            assert docstring.split("Raises:", 1)[1].strip()


@pytest.mark.parametrize("script_name", SCRIPT_PATHS)
def test_scripts_have_main_entry_point(script_name: str) -> None:
    """Check that script entry points are protected by a main guard.

    Args:
        script_name: Key identifying the script.

    Returns:
        None. Assertions verify the expected entry-point structure.
    """
    tree = parse_script(script_name)

    main_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "main"
    ]

    assert len(main_functions) == 1

    main_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
    ]

    assert main_guards


@pytest.mark.parametrize(
    "script_name",
    ["faster_rcnn", "retinanet", "yolo_baseline", "yolo_4class"],
)
def test_training_scripts_have_output_protection(
    script_name: str,
) -> None:
    """Check training scripts for existing-output protection.

    Args:
        script_name: Name of the training script.

    Returns:
        None. Assertions check source-level safety provisions.
    """
    source = get_source(script_name)

    assert "FileExistsError" in source
    assert "output_dir" in source
    assert "best.pt" in source
    assert "reproduction" in source.lower()


def test_faster_rcnn_original_configuration() -> None:
    """Verify the original Faster R-CNN training hyperparameters.

    Returns:
        None. Assertions verify the saved experiment configuration.
    """
    config = get_literal_constants(parse_script("faster_rcnn"))

    assert config["NUM_CLASSES"] == 4
    assert config["NUM_EPOCHS"] == 15
    assert config["BATCH_SIZE"] == 4
    assert config["LEARNING_RATE"] == 0.005
    assert config["MOMENTUM"] == 0.9
    assert config["WEIGHT_DECAY"] == 0.0005
    assert config["STEP_SIZE"] == 5
    assert config["GAMMA"] == 0.1
    assert config["IMAGE_SIZE"] == 640



def test_retinanet_original_configuration() -> None:
    """Verify the five-class and four-class RetinaNet configurations.

    The experiment dictionary contains references to shared class
    names, so its literal configuration values are inspected using
    the abstract syntax tree without executing training code.

    Returns:
        None. Assertions verify the recorded experiment settings.
    """
    tree = parse_script("retinanet")
    config = get_literal_constants(tree)

    assert config["NUM_EPOCHS"] == 15
    assert config["LEARNING_RATE"] == 0.005
    assert config["MOMENTUM"] == 0.9
    assert config["WEIGHT_DECAY"] == 0.0005
    assert config["STEP_SIZE"] == 5
    assert config["GAMMA"] == 0.1

    experiments_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "EXPERIMENTS"
            for target in node.targets
        )
    )

    experiments = {}

    for key_node, value_node in zip(
        experiments_node.keys,
        experiments_node.values,
    ):
        experiment_name = ast.literal_eval(key_node)

        settings = {}

        for setting_key, setting_value in zip(
            value_node.keys,
            value_node.values,
        ):
            name = ast.literal_eval(setting_key)

            try:
                settings[name] = ast.literal_eval(setting_value)
            except (ValueError, TypeError):
                # Class-name constants are checked elsewhere.
                continue

        experiments[experiment_name] = settings

    assert experiments["5class"]["num_classes"] == 5
    assert experiments["4class"]["num_classes"] == 4

    assert experiments["5class"]["batch_size"] == 8
    assert experiments["4class"]["batch_size"] == 4

    assert experiments["5class"]["checkpoint_format"] == "state_dict"
    assert experiments["4class"]["checkpoint_format"] == "full"



def test_yolo_baseline_original_configuration() -> None:
    """Verify both preliminary YOLO training configurations.

    Returns:
        None. Assertions verify baseline and extended settings.
    """
    config = get_literal_constants(parse_script("yolo_baseline"))

    assert config["IMAGE_SIZE"] == 640
    assert config["BATCH_SIZE"] == 16
    assert config["WORKERS"] == 2
    assert config["INITIAL_WEIGHTS"] == "yolov8n.pt"

    experiments = config["EXPERIMENTS"]

    assert experiments["baseline"]["epochs"] == 30
    assert experiments["extended"]["epochs"] == 60


def test_yolo_4class_original_configuration() -> None:
    """Verify the final four-class YOLO training configuration.

    Returns:
        None. Assertions verify the recorded hyperparameters.
    """
    config = get_literal_constants(parse_script("yolo_4class"))

    assert config["EPOCHS"] == 30
    assert config["IMAGE_SIZE"] == 640
    assert config["BATCH_SIZE"] == 16
    assert config["WORKERS"] == 0
    assert config["INITIAL_WEIGHTS"] == "yolov8n.pt"


def test_dataset_preparation_configuration() -> None:
    """Verify the original dataset-cleaning configuration.

    Returns:
        None. Assertions verify dataset names and wrinkle handling.
    """
    config = get_literal_constants(parse_script("prepare_dataset"))

    assert config["ORIGINAL_DATASET"] == "pork_rasher_v4"
    assert config["CLEAN_DATASET"] == "pork_rasher_v4_clean"
    assert config["FOUR_CLASS_DATASET"] == "pork_rasher_v4_4class"

    assert config["WRINKLE_CLASS_ID"] == 4
    assert config["EXPECTED_WRINKLE_SOURCES"] == 6

    assert config["SPLITS"] == ("train", "valid", "test")


def test_dataset_preparation_keeps_source_copies() -> None:
    """Check that preparation uses copies and rejects existing outputs.

    Returns:
        None. Assertions verify source-level dataset safeguards.
    """
    source = get_source("prepare_dataset")

    assert "shutil.copytree" in source
    assert "FileExistsError" in source
    assert "read_yolo_annotations" in source
    assert "remove_wrinkle_images" in source


def test_shared_dataset_loader_remains_separate() -> None:
    """Verify dataset loading and preparation remain separate modules.

    Returns:
        None. Assertions check the current package structure.
    """
    dataset_module = (
        PROJECT_ROOT / "src" / "mma3001" / "dataset.py"
    )

    prepare_module = SCRIPT_PATHS["prepare_dataset"]

    assert dataset_module.is_file()
    assert prepare_module.is_file()
    assert dataset_module != prepare_module
