"""Derive the original step with ONE explicit plant-input hook; no global patch.

The frozen source file is untouched. All control, external-channel, mj_step,
safety, hold and recording statements remain identical. AST equality after
removing this one insertion is tested. This also applies lag during protection.
"""
import ast,inspect,textwrap
from ..disturbance.executor import DisturbanceOwner

def instrumented_step():
    original=DisturbanceOwner.step;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));body=tree.body[0].body
    anchor=[i for i,s in enumerate(body) if isinstance(s,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='last_force' for t in s.targets)]
    if len(anchor)!=1:raise RuntimeError('Frozen executor anchor changed; refuse unknown integration')
    body.insert(anchor[0],ast.parse('force = self.apply_actuator_command(force)').body[0]);ast.fix_missing_locations(tree);environment=dict(original.__globals__);exec(compile(tree,'<phase2c0_frozen_step_plus_one_plant_hook>','exec'),environment)
    return environment['step'],ast.dump(tree,include_attributes=False)

PLANT_STEP,INSTRUMENTED_AST=instrumented_step()
