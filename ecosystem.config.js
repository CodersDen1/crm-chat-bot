module.exports = {
  apps: [
    {
      name: "CRM-Bot",
      script: "bash",
      args: "-c 'source env/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload'",
      watch: true,
      env: {
        PYTHONUNBUFFERED: "1",
        PORT: "8000"
      }
    }
  ]
};