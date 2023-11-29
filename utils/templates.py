from dataclasses import dataclass

TEMPLATE = """{header}
{inputs}

{task}
{test_inputs}

{output_format}
"""

@dataclass
class TemplateConfig:
    header: str = ""
    task: str = ""
    output_format: str = ""

ClassificationTemplate = TemplateConfig(
    header="Here are some text inputs with classification labels, read all of them carefully:",
    task="Now classify these inputs based on the data above:",
    output_format="""
Output format should strictly follow: 
"<input>", Label: <label>
"""
)

MCQArticulationTemplate = TemplateConfig(
    header="",
    task="Now choose out of the following the correct classification rule:",
    output_format="Output should only contain the correct option."
)

default_system = "You are an assistant helping users solve problems."