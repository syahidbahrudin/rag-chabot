FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* ./
RUN npm install --legacy-peer-deps || pnpm install || true
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
