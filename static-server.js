const httpServer = require('http-server');
const path = require('path');

const server = httpServer.createServer({
  root: path.join(__dirname, 'static'),
  cache: 0,
  cors: true
});

server.listen(3000, () => {
  console.log('Static server running on http://localhost:3000');
});