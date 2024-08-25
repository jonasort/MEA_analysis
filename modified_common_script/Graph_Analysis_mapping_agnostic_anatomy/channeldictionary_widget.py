import tkinter as tk
from tkinter import simpledialog, messagebox

class MEAWidget:
    def __init__(self, master):
        self.master = master
        self.master.title("MEA Grid")
        
        self.columns = [chr(i) for i in range(ord('a'), ord('s')) if chr(i) != 'i']
        self.rows = list(range(1, 17))
        
        self.buttons = {}
        self.labels = {}
        self.regions = []
        self.current_region = None
        
        self.is_selecting = False
        self.start_button = None
        
        self.create_left_panel()
        self.create_grid()
        
        self.output_button = tk.Button(master, text="Output Dictionary", command=self.output_dictionary)
        self.output_button.grid(row=17, column=1, columnspan=16)
        
    def create_left_panel(self):
        left_frame = tk.Frame(self.master)
        left_frame.grid(row=0, column=0, rowspan=18, sticky="ns")
        
        add_region_button = tk.Button(left_frame, text="Add Region", command=self.add_region)
        add_region_button.pack(fill=tk.X)
        
        self.region_listbox = tk.Listbox(left_frame)
        self.region_listbox.pack(fill=tk.BOTH, expand=True)
        self.region_listbox.bind('<<ListboxSelect>>', self.on_region_select)
        
    def create_grid(self):
        for i, row in enumerate(self.rows):
            for j, col in enumerate(self.columns):
                if (row == 1 and col in ['a', 'r']) or (row == 16 and col in ['a', 'r']):
                    continue
                button = tk.Button(self.master, text=f"{col.upper()}{row}", width=4, height=2)
                button.grid(row=i, column=j+1)
                button.bind('<Button-1>', self.on_click)
                button.bind('<B1-Motion>', self.on_drag)
                button.bind('<ButtonRelease-1>', self.on_release)
                self.buttons[f"{col.upper()}{row}"] = button
    
    def on_click(self, event):
        if not self.current_region:
            messagebox.showinfo("Info", "Please select a region first.")
            return
        
        self.is_selecting = True
        self.start_button = event.widget
        self.toggle_electrode(event.widget)
    
    def on_drag(self, event):
        if self.is_selecting:
            button = event.widget.winfo_containing(event.x_root, event.y_root)
            if button in self.buttons.values():
                self.toggle_electrode(button, add_only=True)
    
    def on_release(self, event):
        self.is_selecting = False
        self.start_button = None
    
    def toggle_electrode(self, button, add_only=False):
        electrode = button['text'].split('\n')[0]
        
        if not add_only and electrode in self.labels and self.labels[electrode] == self.current_region:
            button['bg'] = 'SystemButtonFace'
            button['text'] = electrode
            del self.labels[electrode]
        else:
            button['bg'] = 'yellow'
            button['text'] = f"{electrode}\n{self.current_region}"
            self.labels[electrode] = self.current_region
    
    def add_region(self):
        region = simpledialog.askstring("Input", "Enter new region name:")
        if region and region not in self.regions:
            self.regions.append(region)
            self.region_listbox.insert(tk.END, region)
    
    def on_region_select(self, event):
        selection = self.region_listbox.curselection()
        if selection:
            self.current_region = self.region_listbox.get(selection[0])
    
    def output_dictionary(self):
        output = {}
        for electrode, region in self.labels.items():
            if region not in output:
                output[region] = []
            output[region].append(electrode)
        
        print(output)
        messagebox.showinfo("Output", str(output))

root = tk.Tk()
mea_widget = MEAWidget(root)
root.mainloop()