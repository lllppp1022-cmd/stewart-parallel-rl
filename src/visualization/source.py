"""Read-only reuse of the archived proxy model, without regenerating old XML."""
import json,itertools
import mujoco
from ..model import ROOT
from ..phase1.collision import CollisionQuery
class ReadOnlyProxy(CollisionQuery):
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'models/phase1_proxy_query.xml'));self.data=mujoco.MjData(self.model)
        self.description=json.loads((ROOT/'results/phase1_collision_audit.json').read_text(encoding='utf-8'))
        excluded={frozenset([x['body1'],x['body2']]) for x in self.description['exclusions']};ids=[i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith('proxy_')];self.pairs=[]
        for a,b in itertools.combinations(ids,2):
            ba=self.model.body(int(self.model.geom_bodyid[a])).name;bb=self.model.body(int(self.model.geom_bodyid[b])).name
            if ba!=bb and frozenset([ba,bb]) not in excluded:self.pairs.append((a,b))
