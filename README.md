# Deep SQL Guide 🚀

### 🔧 Slurm Setup: Connecting to GPU Node

Before running any code, you need to connect to the GPU cluster and activate the appropriate environment.

##### 1
 Connect to the GPU Node
```bash
ssh gpu
```
#### Request a GPU with Slurm
```bash
srun -p hard --gpus-per-node=1 --constraint=A6000 --pty bash
```
📌 Explanation:
	•	srun → Launches a Slurm job interactively.
	•	-p hard → Specifies the partition (hard).
	•	--gpus-per-node=1 → Requests 1 GPU.
	•	--constraint=A6000 → Ensures allocation of an A6000 GPU (you can try A5000).
	•	--pty bash → Starts an interactive bash session.

#### Activate the Conda Environment
```bash
conda activate [env]
```
🔹 Replace [env] with the name of your Conda environment.

⸻

### 🖥️ Starting Ollama

Ollama is required for the model. Start the Ollama server in the background:
```bash
ollama serve &
```

🔹 The & runs the server in the background so you can continue using the terminal.

⸻

### 🚀 Running the Code

Now you’re ready to run Deep SQL!

🔹 Running the Script Interactively

If you want to monitor execution in real-time, run:

```bash
cd deep_sql/scripts
bash train/run.sh
```

⸻

☕ Running the Script in the Background (Job Submission)

If you don’t need real-time output and prefer to let it run in the background, submit it as a batch job:
```bash
cd deep_sql/scripts
sbatch train/run.sh
```
🔹 This submits the script to Slurm and frees your terminal for other tasks.
