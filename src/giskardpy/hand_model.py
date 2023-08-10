class Hand(dict):
    def __init__(self, hand_tool_frame, palm_link, thumb, fingers, opening_width, finger_js):
        dict.__init__(self,
                      hand_tool_frame=hand_tool_frame,
                      palm_link=palm_link,
                      thumb=thumb,
                      fingers=fingers,
                      opening_width=opening_width,
                      finger_js=finger_js)


class Finger(dict):
    def __init__(self, tip_tool_frame, collision_links):
        dict.__init__(self,
                      tip_tool_frame=tip_tool_frame,
                      collision_links=collision_links)

