from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_agent_middleware.config import Settings
from grasp_agent_middleware.langchain_adapter import LangChainToolAdapter
from grasp_agent_middleware.orchestration import build_workflow
from grasp_agent_middleware.schemas import ImageInput, PipelineRequest
from grasp_agent_middleware.visualization import object_table_overlay_svg


class PipelineContractTest(unittest.TestCase):
    def test_pipeline_emits_object_table_and_plan(self) -> None:
        workflow = build_workflow(Settings())
        response = workflow.run(
            PipelineRequest(
                image=ImageInput(image_path="samples/images/demo_sort_scene.png"),
                instruction="Sort the red cube to bin A and the blue cylinder to bin B.",
                request_id="test_request",
            )
        )

        self.assertGreaterEqual(len(response.object_table.objects), 2)
        object_ids = {obj.object_id for obj in response.object_table.objects}
        step_ids = {step.object_id for step in response.plan.steps if step.object_id}
        self.assertTrue(step_ids <= object_ids)
        command_ids = {command.object_id for command in response.execution_commands if command.object_id}
        self.assertTrue(command_ids <= object_ids)
        self.assertGreaterEqual(len(response.agent_trace), 6)
        self.assertGreaterEqual(len(response.execution_commands), len(response.plan.steps))
        self.assertIsNotNone(response.dry_run)
        self.assertIn("object_table", response.artifacts)
        self.assertIn("plan", response.artifacts)
        self.assertIn("execution_commands", response.artifacts)
        self.assertIn("agent_trace", response.artifacts)

        overlay = object_table_overlay_svg(response.object_table)
        self.assertIn("<svg", overlay)
        self.assertIn("obj_001", overlay)

    def test_camera_input_and_langchain_manifest_contracts(self) -> None:
        camera = ImageInput(camera_index=0)
        self.assertEqual(camera.display_ref(), "camera:0")

        workflow = build_workflow(Settings())
        manifest = LangChainToolAdapter(workflow).manifest()
        self.assertEqual(manifest.integration, "langchain")
        self.assertGreaterEqual(len(manifest.tools), 5)
        self.assertIn("robotic_sorting_pipeline", {tool["name"] for tool in manifest.tools})


if __name__ == "__main__":
    unittest.main()
