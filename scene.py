from manim import *
from Two_states_model import mouse0, mouse1
import numpy as np

class ActionNodes(Scene):
    def construct(self):
        
        lever1 = Circle(radius=0.5, color=BLUE, fill_opacity=0.7)
        lever1.move_to(DOWN*1 + RIGHT*1)
        lever1_text = Text("LEVER 1", font_size=18)
        lever1_text.move_to(lever1.get_center())

        lever2 = Circle(radius=0.5, color=PINK, fill_opacity=0.7)
        lever2.move_to(UP*1 + RIGHT*1)
        lever2_text = Text("LEVER 2", font_size=18)
        lever2_text.move_to(lever2.get_center())
        
        idle = Circle(radius=0.5, color=WHITE, fill_opacity=0.7)
        idle.move_to(RIGHT*3)
        idle_text = Text("IDLE", font_size=18)
        idle_text.move_to(idle.get_center())

        # Display reward counter as an updater
        r = 0
        rewards = Text(f"Rewards {r}", font_size=18)
        rewards.move_to(UP*2.5)

        def update_rewards(mob):
            mob.become(Text(f"Rewards {r}", font_size=18))
            mob.move_to(UP*2.5)

        rewards.add_updater(update_rewards)

        # Draw partition line
        top_partition = [0, 1, 0]
        bot_partition = [0, -1, 0]
        partition = Line(top_partition, bot_partition, color=WHITE)
        
        # Add mobjects to scene 
        self.add(
            idle, lever1, lever2, lever1_text, 
            lever2_text, idle_text, rewards, partition
        )
        self.wait(1)
        
        def highlight_node(mob):
            self.play(mob.animate.set_stroke(GREEN, width=6, opacity=1))
            self.wait(0.5)
            self.play(mob.animate.set_stroke(opacity=0))

        # Map states to mobjects
        ACT_TO_NODE = {
            np.nan: idle,
            0: lever1,
            1: lever2,
        }
            
        nodes = lever1, lever2, idle
        n_nodes = len(nodes)    

        for i in range(n_nodes-1):
            for j in range(i+1, n_nodes):
                start_point = self.get_tangent_point(nodes[i], nodes[j])
                end_point = self.get_tangent_point(nodes[j], nodes[i])
                edge = Line(start_point, end_point)
                self.add(edge)

        highlight_node(idle)
        """
        for state in mouse0.action_history[0: 3]:
            mob = ACT_TO_NODE[state]
            highlight_node(mob)
        """

    def get_tangent_point(self, node1, node2):
        """Find the tangent point on the circumference of node1 towards node2."""
        center1 = node1.get_center()
        center2 = node2.get_center()
        direction = normalize(center2 - center1)  # Direction vector from node1 to node2
        return center1 + direction * node1.radius

print(mouse1.action_history)