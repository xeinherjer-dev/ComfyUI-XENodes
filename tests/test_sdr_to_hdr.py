import unittest
import asyncio
import torch
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.sdr_to_hdr import (
    SDRtoHDR,
    SDRtoHDRExtension,
    comfy_entrypoint as sdr_to_hdr_entrypoint,
)
from custom_nodes.xenodes.nodes.sdr_to_hdr_using_model import (
    LoadSDRtoHDRModel,
    SDRtoHDRUsingModel,
    SDRtoHDRAI,
    SDRtoHDRUsingModelExtension,
    comfy_entrypoint as sdr_to_hdr_model_entrypoint,
)


class TestSDRtoHDRNodes(unittest.TestCase):
    def test_sdr_to_hdr_schema(self):
        schema = SDRtoHDR.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SDRtoHDR")
        self.assertEqual(len(schema.outputs), 2)

    def test_sdr_to_hdr_execute_image(self):
        img_tensor = conftest_base.create_dummy_image(batch=1, height=16, width=16)

        # Execute algorithmic HDR expansion (PQ)
        out_pq = SDRtoHDR.execute(
            images=img_tensor,
            peak_nits=400.0,
            itm_knee=0.0,
            itm_exponent=1.0,
            color_space="HDR PQ",
        )
        hdr_images = out_pq.result[0]
        self.assertIsInstance(hdr_images, torch.Tensor)
        self.assertEqual(hdr_images.shape, img_tensor.shape)
        # Check that color_space metadata is set on the tensor
        self.assertEqual(getattr(hdr_images, "_color_space", None), "HDR PQ")

        # Execute algorithmic HDR expansion (HLG)
        out_hlg = SDRtoHDR.execute(
            images=img_tensor,
            peak_nits=400.0,
            color_space="HDR (HLG)",
        )
        hdr_hlg_images = out_hlg.result[0]
        self.assertEqual(getattr(hdr_hlg_images, "_color_space", None), "HDR")

    def test_sdr_to_hdr_model_schemas(self):
        # Load model schema
        load_schema = LoadSDRtoHDRModel.define_schema()
        self.assertEqual(load_schema.node_id, "XENodes.LoadSDRtoHDRModel")
        self.assertTrue(load_schema.is_experimental)

        # Model execution schema
        exec_schema = SDRtoHDRUsingModel.define_schema()
        self.assertEqual(exec_schema.node_id, "XENodes.SDRtoHDRUsingModel")
        self.assertTrue(exec_schema.is_experimental)

        # Deprecated alias schema
        ai_schema = SDRtoHDRAI.define_schema()
        self.assertEqual(ai_schema.node_id, "XENodes.SDRtoHDRAI")
        self.assertTrue(ai_schema.is_deprecated)

    def test_extensions(self):
        ext_algo = asyncio.run(sdr_to_hdr_entrypoint())
        self.assertIsInstance(ext_algo, SDRtoHDRExtension)
        nodes_algo = asyncio.run(ext_algo.get_node_list())
        self.assertIn(SDRtoHDR, nodes_algo)

        ext_model = asyncio.run(sdr_to_hdr_model_entrypoint())
        self.assertIsInstance(ext_model, SDRtoHDRUsingModelExtension)
        nodes_model = asyncio.run(ext_model.get_node_list())
        self.assertIn(LoadSDRtoHDRModel, nodes_model)
        self.assertIn(SDRtoHDRUsingModel, nodes_model)
        self.assertIn(SDRtoHDRAI, nodes_model)


if __name__ == "__main__":
    unittest.main()
