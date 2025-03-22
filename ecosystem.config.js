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
    },
    {
      name: 'CRM-Bot-Static',
      script: './static-server.js',
      instances: 1,
      autorestart: true,
      watch: false
    }
  ]
};