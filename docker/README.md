# docker/

Docker build contexts for each service.

| Folder | Service | Purpose |
|---|---|---|
| `showdown/` | `showdown` | Local Pokémon Showdown server (Node.js) |
| `trainer/` | `trainer` | Python training environment (PyTorch, poke-env, sb3-contrib) |
| `scraper/` | `scraper` | Lightweight Python environment for the Pikalytics scraper |

## Services

```bash
docker-compose up                  # start showdown + tensorboard
docker-compose run --rm trainer …  # run a one-off training command
docker-compose run --rm scraper    # run the Pikalytics scraper
```

TensorBoard is available at `http://localhost:6006` when running `docker-compose up`.
