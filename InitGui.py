import FreeCAD
import FreeCADGui
import os
from ElexusCADMain import ElexusCADInterface

# Define ElexusCommand FIRST
class ElexusCommand:
    """Creates a command to open the Elexus dialog"""

    def GetResources(self):
        icon_path = os.path.join(os.getenv('APPDATA'), "FreeCAD", "Mod", "Elexus", "Resources", "icon.png")
        if not os.path.exists(icon_path):
            FreeCAD.Console.PrintError(f"Icon not found: {icon_path}\n")
        else:
            FreeCAD.Console.PrintMessage(f"Icon found: {icon_path}\n")
        return {
            "Pixmap": icon_path,
            "MenuText": "Open Elexus",
            "ToolTip": "Launch the Elexus text-to-CAD tool"
        }

    def Activated(self):
        """Function that runs when the button is clicked"""
        FreeCAD.Console.PrintMessage("ElexusCommand Activated\n")
        dialog = ElexusCADInterface()
        dialog.exec_()

    def IsActive(self):
        return True  # Always active

# Register the command
FreeCAD.Console.PrintMessage("Registering Elexus_OpenDialog command\n")
FreeCADGui.addCommand("Elexus_OpenDialog", ElexusCommand())

# Now Define the Workbench
class ElexusWorkbench(FreeCADGui.Workbench):
    """Defines the Elexus Workbench"""

    MenuText = "Elexus"
    ToolTip = "Elexus Workbench - Generate CAD from text prompts"
    Icon = os.path.join(os.getenv('APPDATA'), "FreeCAD", "Mod", "Elexus", "Resources", "icon.png")

    def Initialize(self):
        """Setup workbench toolbar"""
        FreeCAD.Console.PrintMessage("Initializing Elexus Workbench...\n")
        self.appendToolbar("Elexus Tools", ["Elexus_OpenDialog"])
        self.appendMenu("Elexus", ["Elexus_OpenDialog"])
        FreeCAD.Console.PrintMessage("Elexus Workbench initialized.\n")

    def GetClassName(self):
        return "Gui::PythonWorkbench"

FreeCAD.Console.PrintMessage("Registering ElexusWorkbench\n")
FreeCADGui.addWorkbench(ElexusWorkbench())  # Register the workbench
FreeCAD.Console.PrintMessage("Elexus Workbench registered.\n")

# Ensure the code is executed
if __name__ == "__main__":
    FreeCAD.Console.PrintMessage("Executing InitGUI.py\n")