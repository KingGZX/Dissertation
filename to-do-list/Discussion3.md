1. Learn what is "Multi-Scale"
2. Extract some gait cycles from one person and use vote to determine the final label of this person. (Each gait cycle can output a label then choose the majority as the final label)
3. Build a Multi-Head model based on the baseline model to perform joint learning.
4. L1Loss
5. padding issues.  (huge difference of the spent time on one gait cycle between different patients which hinder batch training)