# Build the static bundle, then serve it with nginx. nginx also forwards API traffic to the
# api service, so the browser sees a single origin — the session cookie relies on that.
FROM node:20-slim AS build
WORKDIR /srv
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /srv/dist /usr/share/nginx/html
EXPOSE 80
