from __future__ import annotations

import unittest

from jarvis.identity import PRODUCT_NAME, PRODUCT_SLUG, VERSION, user_agent


class ProductIdentityTest(unittest.TestCase):
    def test_public_identity_is_zestoles(self):
        self.assertEqual(PRODUCT_NAME, "ZESTOLES")
        self.assertEqual(PRODUCT_SLUG, "zestoles")

    def test_http_identity_is_ascii_and_versioned(self):
        value = user_agent()
        self.assertIn(PRODUCT_NAME, value)
        self.assertIn(VERSION, value)
        value.encode("ascii")


if __name__ == "__main__":
    unittest.main()
