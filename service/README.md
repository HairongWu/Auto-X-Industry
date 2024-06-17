# Auto-X Service

Auto-X Service provides realtime interactive autonomous services via Auto-X AI server.

## Quickstart

1. Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop) installed (v18+) and started. 
2. Download [OpenRemote](https://github.com/openremote/openremote) and put in this folder. 
3. Install JDK17
4. run CMD:
```
./gradlew clean installDist
docker-compose -p openremote -f profile/dev-ui.yml up --build -d
```
5. If all goes well then you should now be able to access the OpenRemote Manager UI at https://localhost. 
   The default login is username 'admin' with password 'secret'.
