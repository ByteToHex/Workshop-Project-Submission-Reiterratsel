# SystemCode

Self-contained Docker runnable copy of the `REITterratsel` submission setup.

## App-only mode

```powershell
docker compose up --build
```

Open `http://localhost:8501`.

## Rebuild mode

```powershell
docker compose --profile rebuild up --build
```

This starts Neo4j, mounts `docker-compose.env` to `/.env` inside the rebuild container, rebuilds the annual label, CAR-path, and Mamdani cache outputs, and then the app can be started from the same folder.
