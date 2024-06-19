# Auto-X Service

Auto-X Service provides interactive autonomous services for practical industrial settings.

## Auto IoT Solution

### Agents
- Auto-X AI Server
- Eclipse Ditto
- Auto-X Studio
- Auto-X ERP

### Asset List
- Building Asset
- City Asset
- Parking Asset
- Room Asset
- Ship Asset

- Electric Vehicle Asset
- Electricity Asset (Provider and Consumer)
- Gas Asset (Provider and Consumer)
- Water Asset (Provider and Consumer)
- Gauge Asset
- Light Asset
- PV Solar Asset
- Wind Turbine Asset
- Weather Asset
- Environment Sensor Asset

- People Counter Asset
- Drone Asset

### Widgets for Visualization
- Image
- Line Chart
- Gauge
- Table
- Map
- Gateway
- Attribute
- KPI
- Report
- Drone Mission Planner

### Quick Start
1. Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop) installed (v18+) and started. 
2. Install JDK17
3. change to the 'iot' folder and run CMD:
```
./gradlew clean installDist
docker-compose -p openremote -f profile/dev-ui.yml up --build -d
```
4. If all goes well then you should now be able to access the OpenRemote Manager UI at https://localhost. 
   The default login is username 'admin' with password 'secret'.

### Tutorials


## Auto System Development

### Features
- integrate with Auto-X AI Server
- import legacy systems and generate technical documets
- regenerate systems using technical documents
- auto building, unit testing and bug reporting
- initial codebase with technical documets

Change to 'development' folder and refer to [here](https://github.com/JetBrains/intellij-community) to setup the build envireonment.

## Auto ERP
  
Please refer to [here](https://www.odoo.com/documentation/master/administration/on_premise/source.html) to run Odoo locally.

### Auto Supply Chain

- integrate with Auto-X IoT
- integrate with Auto-X AI Server
- demand forecasting
- inventory optimization
- production planning

1. Make sure you have [PostgreSQL](https://www.postgresql.org/download/) installed. 
2. Change to the 'erp/supply_chain' folder and run command
```
mkdir build
cd build
cmake --build . --config Release
```


## Auto Clinic

### Features

- integrate with Auto-X Studio
- integrate with Auto-X AI Server
- integrated with Auto ERP (Pharmacy, Laboratory, OPD, Patient billing, Therapy, Appointment management)
- WebRTC and digital human based auto admission/diagnosis
- auto-generated comprehensive therapy advises
- Face identification using Digital Persona library
- Vaccines database
- 3BT clinical thesaurus with validated coding aid for ICD-10 and ICPC-2
- Snomed CT coding (diagnoses) starting from version 4.0
- Full lab order entry and results management (LOINC coding supported). 
- X-ray and pathology results management
- Multimedia (pictures, video, audio) support
- Integrates RxNorm based multi-lingual drug-drug interaction detection
- Integrated solution for archiving of scanned documents
- Integration of DCM4CHE/Weasis based PACS and DICOM-viewer solution
- HL7/FHIR API for structured data exchange with external applications

