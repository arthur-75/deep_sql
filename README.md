
# deep_sql



Slurm commands at start :

```bash
ssh gpu
srun -p hard --gpus-per-node=1 --constraint=A6000 --pty bash
conda activat [env]
```

start Ollama 

```bash
ollama serve &
```

run the code

```bash
cd deep_sql/scripts
bash train/run.sh
```

run the code and go take a cofee

```bash
cd deep_sql/scripts
sbatch train/run.sh
```

