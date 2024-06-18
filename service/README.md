# Auto-X Service

Auto-X Service provides interactive autonomous services for practical industrial settings.

## Auto IoT Solution
### Agents
- Auto-X AI Server
- Eclipse Ditto
- Auto-X AI Studio
- Auto-X ERP
- Auto-X Supply Chain (eg. fleet mangement)

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
- Mission Planner

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

## Auto ERP

## Auto Health Care

## Auto Supply Chain

## Auto Finance