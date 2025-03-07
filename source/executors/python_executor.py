import subprocess

def execute_python_code(code: str) -> dict:
    """
    Exécute un morceau de code Python et retourne le résultat.

    :param code: Code Python sous forme de chaîne de caractères.
    :return: Dictionnaire contenant le statut et le résultat.
    """
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
        return {"success": True, "output": result.stdout}
    except Exception as e:
        return {"success": False, "error": str(e)}