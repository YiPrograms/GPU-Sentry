import hashlib
import tempfile
import unittest
from pathlib import Path

from gpu_sentry.sass.ingest import IngestError, copy_code_objects


class CodeIdentityTest(unittest.TestCase):
    def test_code_id_is_sha256_prefix(self):
        data = b"test cubin"
        digest = hashlib.sha256(data).hexdigest()
        code_id = int(digest[:16], 16)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            workload = root / "workload"
            (capture / "code").mkdir(parents=True)
            source = capture / "code" / f"code_{code_id:016x}.bin"
            source.write_bytes(data)

            event = {
                "code_id": code_id,
                "code_type": 2,
                "sha256": digest,
                "size": len(data),
                "path": f"code/{source.name}",
            }
            code_map = copy_code_objects(capture, [event], workload)
            self.assertIn(str(code_id), code_map)

            event["code_id"] += 1
            with self.assertRaises(IngestError):
                copy_code_objects(capture, [event], workload)


if __name__ == "__main__":
    unittest.main()
