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
    task="Now choose the correct classification rule out of the following :",
    output_format="Output should only contain the correct option."
)

default_system = "You are an assistant helping users solve problems."

FreeformArticulationTemplate = TemplateConfig(
    header="",
    task="Now state the classification rule used to classify the inputs above.",
    output_format="You can think step by step in the output."
)

FreeformEvaluationTemplate = TemplateConfig(
    header="Read the following descriptive answer carefully:",
    task="Now choose the option closest to the descriptive answer above out of the following:",
    output_format="Output should only contain the correct option."
)

TrueFalseEvaluationTemplate = TemplateConfig(
    header = "Read the following descriptive answer carefully:",
    task = "You need to check if the descriptive answer is correct or not. The correct answer is the following:",
    output_format = "You must respond True if the descriptive answer is correct, False otherwise. You can think step by step in the output, but only your final answer will be taken. So you need to finally output 'True' or 'False'."
)