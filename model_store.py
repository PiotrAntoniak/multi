"""Small persistence layer for trained XGBoost model bundles.

Models are LOCAL artifacts: bundles live under ``models/<name>/`` and are
git-ignored, never committed.  The format is XGBoost's native binary (``.ubj``);
no pickle is used, so bundles are portable across Python versions and only
depend on the XGBoost version recorded in the manifest.

Bundle layout::

    models/<name>/tag_<j>.ubj      one file per tag (``XGBClassifier.save_model``)
    models/<name>/manifest.json    ``meta`` + name / n_tags / created / xgboost

The public API is ``save_bundle`` / ``load_bundle`` / ``list_bundles``.
"""
import json
import os
import time

import xgboost

MANIFEST = "manifest.json"


def _bundle_dir(name, root):
    return os.path.join(root, name)


def _manifest_path(name, root):
    return os.path.join(_bundle_dir(name, root), MANIFEST)


def save_bundle(name, models, meta, root="models"):
    """Write ``models`` (list of fitted XGBClassifier) and ``meta`` to ``models/<name>/``.

    Each classifier ``j`` is written to ``tag_<j>.ubj`` with its native
    ``save_model``; the manifest is ``meta`` extended with ``name``,
    ``n_tags``, an ISO ``created`` timestamp and the ``xgboost`` version.
    Existing files for the same name are overwritten.  Returns the bundle dir.
    """
    outdir = _bundle_dir(name, root)
    os.makedirs(outdir, exist_ok=True)
    for j, model in enumerate(models):
        model.save_model(os.path.join(outdir, "tag_%d.ubj" % j))
    manifest = dict(meta)
    manifest.update({
        "name": name,
        "n_tags": len(models),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "xgboost": xgboost.__version__,
    })
    mpath = _manifest_path(name, root)
    tmp = mpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, mpath)
    return outdir


def load_bundle(name, root="models"):
    """Load ``models/<name>/`` -> ``(models, meta)``.

    ``models`` is a list of XGBClassifier objects in manifest tag-index order;
    ``meta`` is the decoded manifest dict.  Raises ``FileNotFoundError`` if the
    bundle manifest is missing and ``ValueError`` if a declared tag file is
    absent (tag-count mismatch).
    """
    mpath = _manifest_path(name, root)
    if not os.path.isfile(mpath):
        raise FileNotFoundError("model bundle not found: %s" % mpath)
    with open(mpath, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta.get("n_tags", -1))
    if n < 0:
        raise ValueError("bundle %r manifest has no valid n_tags" % name)
    models = []
    for j in range(n):
        fpath = os.path.join(_bundle_dir(name, root), "tag_%d.ubj" % j)
        if not os.path.isfile(fpath):
            raise ValueError(
                "bundle %r manifest declares n_tags=%d but %s is missing"
                % (name, n, fpath))
        model = xgboost.XGBClassifier()
        model.load_model(fpath)
        models.append(model)
    return models, meta


def list_bundles(root="models"):
    """Sorted names of the bundles (dirs holding a manifest) under ``root``."""
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, MANIFEST)))
