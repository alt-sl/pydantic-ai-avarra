from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
import yaml
from yaml import SafeLoader
import os
import logfire
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits



# Configure logfire
logfire.configure(send_to_logfire='if-token-present')

class EditRequest(BaseModel):
    """Request to edit a specific section of the simulation."""
    section: Literal["thoughts", "memories", "communication_guidelines"]
    edit_instructions: str

class ThoughtsEdit(BaseModel):
    """The edited thoughts section of the simulation."""
    thoughts: List[str]

# Main Sim Editor Agent
sim_editor_agent = Agent[None, EditRequest](
    'claude-3-5-sonnet-latest',
    result_type=EditRequest,
    system_prompt=(
        'You are an expert simulation editor agent. Your job is to analyze edit requests '
        'and determine which section needs modification. Currently, you can only handle '
        'thoughts section edits. When a request comes in, determine if it relates to thoughts '
        'and create an appropriate EditRequest.'
    ),
)

# Specialized Thoughts Editor Agent
thoughts_editor_agent = Agent[None, ThoughtsEdit](
    'claude-3-5-sonnet-latest',
    result_type=ThoughtsEdit,
    system_prompt=(
        'You are an expert thoughts editor. Your job is to edit the thoughts section '
        'of simulation files while maintaining context and personality. Format each '
        'thought as a clear, complete statement.'
    ),
)

@sim_editor_agent.tool
async def edit_thoughts(ctx: RunContext[None], request: EditRequest) -> str:
    """Edit the thoughts section using the specialized thoughts editor."""
    if request.section != "thoughts":
        return "Can only handle thoughts editing currently."
    
    result = await thoughts_editor_agent.run(
        request.edit_instructions,
        usage=ctx.usage,
        usage_limits=UsageLimits(request_limit=5)
    )
    
    # Update the YAML file with new thoughts
    with open('sim.yaml', 'r') as f:
        sim_data = yaml.load(f, Loader=SafeLoader)
    
    prompt = sim_data['sim']['elements'][0]['prompt']
    start = prompt.find('Your thoughts as you join the call')
    end = prompt.find('Your recent memories for context')
    
    thoughts_text = 'Your thoughts as you join the call (in no particular order):\n'
    for thought in result.data.thoughts:
        thoughts_text += f'{thought}\n\n'
    
    new_prompt = prompt[:start] + thoughts_text + prompt[end:]
    sim_data['sim']['elements'][0]['prompt'] = new_prompt
    
    with open('sim.yaml', 'w') as f:
        yaml.dump(sim_data, f, allow_unicode=True)
    
    return f"Successfully updated thoughts section with {len(result.data.thoughts)} thoughts."

async def main():
    usage = Usage()
    usage_limits = UsageLimits(request_limit=15)
    
    while True:
        edit_request = Prompt.ask("What would you like to edit? (or 'quit' to exit)")
        if edit_request.lower() == 'quit':
            break
            
        result = await sim_editor_agent.run(
            edit_request,
            usage=usage,
            usage_limits=usage_limits
        )
        
        if isinstance(result.data, EditRequest):
            response = await edit_thoughts(RunContext(None, result.model, usage, edit_request), result.data)
            print(response)
        else:
            print("Failed to process edit request")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())