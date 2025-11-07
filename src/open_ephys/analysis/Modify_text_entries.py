import numpy as np

file_path = r"ABATE-008_2025-05-14_14-34-09_001\Record Node 106\experiment1\recording1\events\MessageCenter\text.npy"
texts = np.load(file_path, allow_pickle=True)

decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]

'''# Print and modify
for i, msg in enumerate(decoded_texts):
    print(f"{i}: {msg}")
decoded_texts[91] = "starting 0.75 mA"
decoded_texts[98] = "starting 0.50 mA"
decoded_texts[104] = "starting 0.40 mA"
decoded_texts[115] = "starting 0.20 mA"
decoded_texts[121] = "starting 0.10 mA"
decoded_texts[127] = "starting 0.08 mA"
decoded_texts[133] = "starting 0.06 mA"
decoded_texts[140] = "starting 0.04 mA"
decoded_texts[146] = "starting 0.02 mA"
decoded_texts[152] = "starting 0.01 mA"
decoded_texts[154] = "starting 1.0 mA"




# Save modified
encoded_texts = np.array([t.encode('utf-8') for t in decoded_texts], dtype=object)
np.save("text_modified.npy", encoded_texts)

#---------------


file_path = r"text_modified.npy"
texts = np.load(file_path, allow_pickle=True)

decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]

# Print and modify
for i, msg in enumerate(decoded_texts):
    print(f"{i}: {msg}")

'''