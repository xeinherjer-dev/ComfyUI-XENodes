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
        input_ids = [inp.id for inp in exec_schema.inputs]
        self.assertIn("exposure", input_ids)
        self.assertIn("lift", input_ids)
        exposure_inp = next(inp for inp in exec_schema.inputs if inp.id == "exposure")
        lift_inp = next(inp for inp in exec_schema.inputs if inp.id == "lift")
        self.assertEqual(exposure_inp.default, 0.0)
        self.assertEqual(lift_inp.default, 0.0)

    def test_hdr_luminance_adjustments(self):
        from custom_nodes.xenodes.utils.color import adjust_hdr_luminance, pq_to_hdr_output

        # Create dummy linear BT.2020 image (1.0 = 100 nits)
        linear = torch.tensor([[[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]]], dtype=torch.float32)

        # 1. Neutral (no change when exposure=0 and lift=0)
        unchanged = adjust_hdr_luminance(linear, exposure=0.0, lift=0.0)
        torch.testing.assert_close(unchanged, linear)

        # 2. Exposure (+1 EV should double luminance)
        doubled = adjust_hdr_luminance(linear, exposure=1.0, lift=0.0)
        torch.testing.assert_close(doubled, linear * 2.0, rtol=1e-4, atol=1e-4)

        # 3. Lift should raise shadow/midtone
        lifted = adjust_hdr_luminance(linear, exposure=0.0, lift=0.1)
        self.assertTrue(torch.all(lifted >= linear))
        # Shadow pixel [0.1, 0.2, 0.3] should be raised more proportionally than [1.0, 1.0, 1.0]
        shadow_gain = (lifted[0, 0, 0] / linear[0, 0, 0]).item()
        white_gain = (lifted[0, 1, 0] / linear[0, 1, 0]).item()
        self.assertGreater(shadow_gain, white_gain)

        # 4. pq_to_hdr_output integration
        dummy_pq = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
        out_hdr = pq_to_hdr_output(dummy_pq, color_space="HDR PQ", peak_nits=1000.0, exposure=0.5, lift=0.05)
        self.assertEqual(out_hdr.shape, dummy_pq.shape)
        self.assertTrue(torch.all(out_hdr >= 0.0) and torch.all(out_hdr <= 1.0))

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

    def test_hg_and_hr_support(self):
        from custom_nodes.xenodes.utils.hdr_models import (
            DEFAULT_MODELS,
            HallucinationGenerator,
            generate_highlight_mask,
            run_sdr_to_hdr_frame,
            SDRtoHDRModelBundle,
            ARCH_HG,
        )

        # 1. HR.pth is in default models list
        self.assertIn("HR.pth", DEFAULT_MODELS)

        # 2. Highlight mask generation
        sdr_test = torch.tensor([[[[0.5, 0.96], [0.98, 1.0]]]], dtype=torch.float32)
        sdr_test = sdr_test.repeat(1, 3, 1, 1)  # (1, 3, 2, 2)
        mask = generate_highlight_mask(sdr_test, threshold=0.95)
        self.assertEqual(mask.shape, (1, 1, 2, 2))
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)  # 0.5 <= 0.95 -> 0.0
        self.assertGreater(mask[0, 0, 0, 1].item(), 0.0)  # 0.96 > 0.95 -> > 0.0
        self.assertEqual(mask[0, 0, 1, 1].item(), 1.0)  # 1.0 -> 1.0

        # 3. HallucinationGenerator forward pass
        hg_net = HallucinationGenerator(nf=16).eval()  # eval mode for BatchNorm
        dummy_in = torch.ones((1, 3, 32, 32), dtype=torch.float32)
        dummy_mask = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        out = hg_net(dummy_in, dummy_mask)
        self.assertEqual(out.shape, dummy_in.shape)
        # When mask is 0, output must equal input (out = mask * out + img)
        torch.testing.assert_close(out, dummy_in)

        # 4. run_sdr_to_hdr_frame with ARCH_HG
        bundle = SDRtoHDRModelBundle(model=hg_net, arch=ARCH_HG, names=["HG_test.pth"])
        res, fea = run_sdr_to_hdr_frame(bundle, dummy_in)
        self.assertEqual(res.shape, dummy_in.shape)
        self.assertIsNone(fea)


if __name__ == "__main__":
    unittest.main()
