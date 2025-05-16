import adsk.core, adsk.fusion, adsk.cam, traceback
import adsk.core, adsk.fusion, adsk.cam, traceback
import os
import time
import subprocess
import sys

def install_kittycad(required_version):
    app = adsk.core.Application.get()
    ui = app.userInterface
    addin_path = os.path.dirname(os.path.realpath(__file__))
    packages_path = os.path.join(addin_path, 'site-packages')
    os.makedirs(packages_path, exist_ok=True)

    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)

    try:
        import kittycad
        print(f"Successfully imported kittycad (version unknown, assuming >= {required_version} is present in site-packages).")
        return
    except ImportError:
        try:
            pip_target_command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"kittycad=={required_version}",
                "--target",
                packages_path,
                "--no-warn-script-location"
            ]
            print(f"Installing kittycad=={required_version} to: {packages_path}")
            subprocess.check_call(pip_target_command)
            import kittycad
            print(f"Successfully installed and imported kittycad {required_version}.")

        except subprocess.CalledProcessError as e:
            ui.messageBox(f"Error installing kittycad: {e}")
        except Exception as e:
            ui.messageBox(f"An unexpected error occurred during kittycad installation: {e}")
    except Exception as e:
        ui.messageBox(f"An unexpected error occurred during initial kittycad import check: {e}")

install_kittycad("0.7.6")
from kittycad.api.ml import create_text_to_cad, get_text_to_cad_model_for_user
from kittycad.client import ClientFromEnv
from kittycad.models import (
    ApiCallStatus,
    Error,
    FileExportFormat,
    TextToCadCreateBody,
)

# Global variables to keep track of UI elements
app = adsk.core.Application.get()
ui = app.userInterface
handlers = []

def create_command():
    command_definitions = ui.commandDefinitions
    cmd_id = 'ElexusCAD' 

    cmd_def = command_definitions.itemById(cmd_id)

    if cmd_def:
        cmd_def.deleteMe()  # Delete old definition if it exists

    if not cmd_def:
        icon_path = os.path.join(os.path.dirname(__file__), "Resources", "TTC")
        
        cmd_def = command_definitions.addButtonDefinition(
            cmd_id, 'ElexusCAD Text-to-CAD', 'Generate CAD from text input \n \n Opens a dialogue box where you can type in a precise description of the object you want to generate including measurements.', icon_path
        )

    class CommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
        def notify(self, args):
            try:
                inputs = args.command.commandInputs
                text_box = inputs.addTextBoxCommandInput('cadPrompt', 'Enter prompt:', '', 5, False)
                text_box.isFullWidth = True

                class CommandExecuteHandler(adsk.core.CommandEventHandler):
                    def notify(self, args):
                        try:
                            prompt = text_box.text if text_box else ''
                            if not prompt:
                                ui.messageBox('Please enter a prompt')
                                return
                            
                            ui.messageBox(f'Submitting prompt: {prompt}')
                            client = ClientFromEnv()
                            response = create_text_to_cad.sync(
                                client=client,
                                output_format=FileExportFormat.STEP,
                                body=TextToCadCreateBody(prompt=prompt),
                            )
                            
                            if isinstance(response, Error) or response is None:
                                ui.messageBox(f'Error: {response}')
                                return
                            
                            result = response
                            while result.completed_at is None:
                                time.sleep(5)
                                result = get_text_to_cad_model_for_user.sync(client=client, id=result.id)
                                if isinstance(result, Error) or result is None:
                                    ui.messageBox(f'Error: {result}')
                                    return
                            
                            if result.status == ApiCallStatus.FAILED:
                                ui.messageBox(f'Text-to-CAD failed: {result.error}')
                                return
                            
                            if not result.outputs or "source.step" not in result.outputs:
                                ui.messageBox("Text-to-CAD completed but no files were returned.")
                                return
                            
                            output_path = os.path.join(os.path.expanduser("~"), "Desktop", "text-to-cad-output.step")
                            try:
                                with open(output_path, "wb") as output_file:
                                    output_file.write(result.outputs["source.step"])
                                
                                ui.messageBox(f'Saved output to {output_path}')
                                
                                import_manager = app.importManager
                                step_options = import_manager.createSTEPImportOptions(output_path)
                                import_manager.importToNewDocument(step_options)

                            except Exception as e:
                                ui.messageBox(f'Error saving file: {e}')
                        except:
                            ui.messageBox('Failed to execute command:\n{}'.format(traceback.format_exc()))

                on_execute = CommandExecuteHandler()
                handlers.append(on_execute)
                args.command.execute.add(on_execute)
            except:
                ui.messageBox('Failed to create command:\n{}'.format(traceback.format_exc()))

    on_created = CommandCreatedHandler()
    handlers.append(on_created)
    cmd_def.commandCreated.add(on_created)

    # **Create a dedicated toolbar panel**
    workspace = ui.workspaces.itemById('FusionSolidEnvironment')
    panel_id = 'ElexusCADPanel'
    
    panel = workspace.toolbarPanels.itemById(panel_id)
    if not panel:
        panel = workspace.toolbarPanels.add(panel_id, 'ElexusCAD', '', False)

    # **Add the button to the ElexusCAD panel**
    panel.controls.addCommand(cmd_def)

    return cmd_def


def run(context):
    try:
        cmd_def = create_command()
    except:
        ui.messageBox('Failed to load plugin:\n{}'.format(traceback.format_exc()))
        
def stop(context):
    try:
        workspace = ui.workspaces.itemById('FusionSolidEnvironment')
        panel_id = 'ElexusCADPanel'
        
        panel = workspace.toolbarPanels.itemById(panel_id)
        if panel:
            panel.deleteMe()
        
        
    except:
        ui.messageBox('Failed to unload plugin:\n{}'.format(traceback.format_exc()))
