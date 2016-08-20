import seaborn as sns

ui_color = [ "#2ecc71", 
             "#dd61f4",
             "#3498db",
             "#e74c3c", 
             "#94c3e4",
             "#b0724c",
             "#68a855",
             "#1c1ae3",
             "#34495e", 
             "#9b59b6" ]

palette = sns.color_palette(ui_color) * 5
palette.insert(0, (0.5,0.5,0.5))
