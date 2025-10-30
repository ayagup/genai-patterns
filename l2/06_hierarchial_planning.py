"""
Pattern 6: Hierarchical Planning
Breaks down goals into hierarchical sub-goals.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class Goal:
    """Represents a goal at any level of hierarchy"""
    id: str
    description: str
    level: str  # 'strategic', 'tactical', 'operational'
    parent_id: Optional[str] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    status: str = 'pending'
    result: Optional[str] = None

class HierarchicalPlanningPattern:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.goals: Dict[str, Goal] = {}
        self.goal_counter = 0
    
    def create_goal(self, description: str, level: str, parent_id: Optional[str] = None) -> str:
        """Create a new goal"""
        goal_id = f"goal_{self.goal_counter}"
        self.goal_counter += 1
        
        goal = Goal(
            id=goal_id,
            description=description,
            level=level,
            parent_id=parent_id
        )
        
        self.goals[goal_id] = goal
        
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(goal)
        
        return goal_id
    
    def decompose_strategic_goal(self, goal: str) -> List[str]:
        """Break strategic goal into tactical sub-goals"""
        template = """Break down this high-level strategic goal into 3-5 mid-level tactical objectives:

Strategic Goal: {goal}

Provide tactical objectives that are more concrete and actionable than the strategic goal,
but still require further breakdown.

Tactical Objectives:"""
        
        prompt = PromptTemplate(template=template, input_variables=["goal"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(goal=goal)
        
        objectives = [obj.strip() for obj in result.split('\n') 
                     if obj.strip() and not obj.strip().startswith('Tactical')]
        return objectives[:5]
    
    def decompose_tactical_goal(self, goal: str) -> List[str]:
        """Break tactical goal into operational tasks"""
        template = """Break down this tactical objective into specific operational tasks:

Tactical Objective: {goal}

Provide 3-7 concrete, executable tasks that will accomplish this objective.

Operational Tasks:"""
        
        prompt = PromptTemplate(template=template, input_variables=["goal"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(goal=goal)
        
        tasks = [task.strip() for task in result.split('\n') 
                if task.strip() and not task.strip().startswith('Operational')]
        return tasks[:7]
    
    def generate_actions(self, task: str) -> List[str]:
        """Generate specific actions for an operational task"""
        template = """For this operational task, list the specific actions to take:

Task: {task}

Provide 2-4 specific, immediate actions.

Actions:"""
        
        prompt = PromptTemplate(template=template, input_variables=["task"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(task=task)
        
        actions = [action.strip() for action in result.split('\n') 
                  if action.strip() and not action.strip().startswith('Actions')]
        return actions[:4]
    
    def plan_hierarchically(self, strategic_goal: str) -> Dict:
        """Create a complete hierarchical plan"""
        # Create strategic level
        strategic_id = self.create_goal(strategic_goal, 'strategic')
        
        # Decompose to tactical level
        tactical_objectives = self.decompose_strategic_goal(strategic_goal)
        tactical_ids = []
        
        for tactical_obj in tactical_objectives:
            tactical_id = self.create_goal(tactical_obj, 'tactical', strategic_id)
            tactical_ids.append(tactical_id)
            
            # Decompose to operational level
            operational_tasks = self.decompose_tactical_goal(tactical_obj)
            
            for op_task in operational_tasks:
                op_id = self.create_goal(op_task, 'operational', tactical_id)
                
                # Generate specific actions
                actions = self.generate_actions(op_task)
                self.goals[op_id].actions = actions
        
        # Build hierarchy summary
        hierarchy = self._build_hierarchy_dict(strategic_id)
        
        return {
            'strategic_goal': strategic_goal,
            'hierarchy': hierarchy,
            'total_goals': len(self.goals),
            'breakdown': {
                'strategic': 1,
                'tactical': len(tactical_ids),
                'operational': sum(len(self.goals[tid].sub_goals) for tid in tactical_ids)
            }
        }
    
    def _build_hierarchy_dict(self, goal_id: str, depth: int = 0) -> Dict:
        """Recursively build hierarchy dictionary"""
        goal = self.goals[goal_id]
        
        result = {
            'id': goal.id,
            'description': goal.description,
            'level': goal.level,
            'depth': depth
        }
        
        if goal.sub_goals:
            result['sub_goals'] = [
                self._build_hierarchy_dict(sg.id, depth + 1)
                for sg in goal.sub_goals
            ]
        
        if goal.actions:
            result['actions'] = goal.actions
        
        return result
    
    def print_hierarchy(self, goal_id: str = None, indent: int = 0):
        """Print the goal hierarchy"""
        if goal_id is None:
            # Find root goals (strategic level)
            roots = [g for g in self.goals.values() if g.parent_id is None]
            for root in roots:
                self.print_hierarchy(root.id)
            return
        
        goal = self.goals[goal_id]
        prefix = "  " * indent
        
        symbol = {
            'strategic': '🎯',
            'tactical': '📋',
            'operational': '✓'
        }.get(goal.level, '•')
        
        print(f"{prefix}{symbol} [{goal.level.upper()}] {goal.description}")
        
        if goal.actions:
            for action in goal.actions:
                print(f"{prefix}    → {action}")
        
        for sub_goal in goal.sub_goals:
            self.print_hierarchy(sub_goal.id, indent + 1)

if __name__ == "__main__":
    hp = HierarchicalPlanningPattern()
    
    strategic_goal = "Launch a successful e-commerce platform within 6 months"
    
    result = hp.plan_hierarchically(strategic_goal)
    
    print("="*70)
    print("HIERARCHICAL PLAN")
    print("="*70)
    print(f"\nStrategic Goal: {result['strategic_goal']}\n")
    print(f"Total Goals: {result['total_goals']}")
    print(f"  Strategic: {result['breakdown']['strategic']}")
    print(f"  Tactical: {result['breakdown']['tactical']}")
    print(f"  Operational: {result['breakdown']['operational']}")
    print("\n" + "="*70)
    print("HIERARCHY BREAKDOWN")
    print("="*70 + "\n")
    
    hp.print_hierarchy()
