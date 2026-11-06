"""Test-suite setup: a private, temporary workspace.

The app keeps projects in ``workspace/`` (examples opened as working copies, the default
project, new studies, the last-project pointer). Tests open examples, save and start studies,
so they get their own folder: RSO_WORKSPACE must be set before ``ui.session`` is imported,
which is why it happens here, at conftest import time. The user's workspace is never touched.
"""

import atexit
import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="rso_test_workspace_")
os.environ["RSO_WORKSPACE"] = _TMP
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
