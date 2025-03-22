module.exports = {
  apps: [
    {
      name: "CRM-Bot",
      script: "bash",
      args: "-c 'source env/bin/activate && uvicorn main:app --host localhost --port 8000 --reload'",
      watch: true,
      env: {
        PYTHONUNBUFFERED: "1",
        PORT: "8000"
      }
    }
  ]
};