<div align="center">
  <img src="assets/logo.png" height="120">
</div>
<div align="center">
  <h1>Auto-X for Industries</h1>
</div>

Not stable for now. Please wait for the first release.

Auto-X for Industries is an autonomous solution that aims to solve the following challenges:

- High running cost and low efficiency for practical industrial uses with current AI services. Customers sometimes defer to use GPUs for some reasons.
- Difficult to annotate/pretrain/finetune for specific scenarios using private datasets with most of current AI services
- Insufficient accuracy and safety for industrial scenarios with current AI services. In some cases, you should use on-premise services only.
- Only part of the business processes can be automated with existing services. A mixed of AI serives can often reduce efficiency and accuracy. 

<div  align="center">
  <img src="assets/framework.png" width="800"/>
</div>

## Release Plans

- [ ] version 0.0.1

1. Auto-X Engine: Big models (llama3, whisper), Tiny models (picodet, tinypose), convert pre-trained models to Auto-X Model File, integrate with Auto-X IoT (MQTT)

2. Auto-X Studio: Built-in solution templates, dataset generation, pre-labeling, fine-tuning/pre-training, knowledge creation from images/documents/websites, integrate with Auto-X Server
   
3. Auto-X Server: integrate with Auto-X Studio and Auto-X IoT
   
4. Recognize anything Demo: recognize objects of images taken by esp32 camera, create image vector database, and update related models

## Other Plans (Automate Existing Business Systems with Auto-X)

- [Auto-X IDE](https://github.com/HairongWu/auto-intellij-community) (based on IntelliJ IDEA Community)

- [Auto-X Search](https://github.com/HairongWu/Auto-X-Search) (based on SWIRL AI Connect)
- [Auto-X RTOS](https://github.com/HairongWu/NuttX-for-AutoX) (based on NuttX)
- [Auto-X ERP](https://github.com/HairongWu/Auto-X-ERP) (based on Odoo)
- [Auto-X Clinic](https://github.com/HairongWu/AutoX-Clinic) (based on OpenClinic GA and Open Hospital)
- Auto Finance (based on FinGPT)
- Auto Insurance (based on openIMIS)
- Auto EDA (based on KiCad)
- Auto CAD (based on FreeCAD)

- Auto-X Consultant (All-In-One)

## Acknowledgments

- [label-studio](https://github.com/HumanSignal/label-studio)
- [openremote](https://github.com/openremote/openremote)
- [IntelliJ IDEA Community Edition](https://github.com/JetBrains/intellij-community)
- [Odoo](https://github.com/odoo/odoo)
- [OpenClinic GA](https://sourceforge.net/projects/open-clinic/)
- [Open Hospital](https://github.com/informatici/openhospital)
- [SWIRL AI Connect](https://github.com/swirlai/swirl-search)