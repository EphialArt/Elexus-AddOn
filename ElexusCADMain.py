import time
from kittycad.api.ml import create_text_to_cad, get_text_to_cad_model_for_user
from kittycad.client import ClientFromEnv
from kittycad.models import (
    ApiCallStatus,
    Error,
    FileExportFormat,
    TextToCadCreateBody,
)
import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets, QtCore
import os
import ImportGui
from freecad import module_io


class ElexusCADInterface(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ElexusCADInterface, self).__init__(parent)
        self.setWindowTitle("ElexusCAD Text-to-CAD")
        self.resize(400, 120)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # Prompt input field
        self.promptEdit = QtWidgets.QLineEdit(self)
        self.promptEdit.setPlaceholderText("Enter your CAD prompt here")
        layout.addWidget(self.promptEdit)

        # Submit button
        self.submitBtn = QtWidgets.QPushButton("Generate Model", self)
        layout.addWidget(self.submitBtn)

        # Status label
        self.statusLabel = QtWidgets.QLabel("", self)
        layout.addWidget(self.statusLabel)

        # Connect button to function
        self.submitBtn.clicked.connect(self.submitPrompt)

    def submitPrompt(self):
        prompt = self.promptEdit.text().strip()
        if not prompt:
            self.statusLabel.setText("Please enter a prompt.")
            return

        self.statusLabel.setText("Submitting prompt to Zoo API...")
        QtWidgets.QApplication.processEvents()

        # Create client
        client = ClientFromEnv()

        # Send the prompt to the API
        response = create_text_to_cad.sync(
            client=client,
            output_format=FileExportFormat.STEP,
            body=TextToCadCreateBody(prompt=prompt),
        )

        if isinstance(response, Error) or response is None:
            self.statusLabel.setText(f"Error: {response}")
            return

        result = response

        # Poll for completion
        while result.completed_at is None:
            time.sleep(5)
            result = get_text_to_cad_model_for_user.sync(client=client, id=result.id)

            if isinstance(result, Error) or result is None:
                self.statusLabel.setText(f"Error: {result}")
                return

        # Check final status
        if result.status == ApiCallStatus.FAILED:
            self.statusLabel.setText(f"Text-to-CAD failed: {result.error}")
            return

        if not result.outputs or "source.step" not in result.outputs:
            self.statusLabel.setText("Text-to-CAD completed but no files were returned.")
            return

        # Save and open file
        output_path = os.path.join(os.path.expanduser("~"), "Desktop", "text-to-cad-output.step")

        # Save the file as binary
        with open(output_path, "wb") as output_file:
            output_file.write(result.outputs["source.step"])  # Extract the file content

        self.statusLabel.setText(f"Saved output to {output_path}")
        module_io.OpenInsertObject("ImportGui", output_path, "insert", FreeCAD.ActiveDocument.Name)

# Run the interface in FreeCAD
dialog = ElexusCADInterface()
dialog.show()
