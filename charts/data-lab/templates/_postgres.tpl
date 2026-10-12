{{/*
Single-instance Postgres used as a component's private database.
Usage: include "datalab.postgres" (list $ "hms-db" .Values.hms.database (include "datalab.hmsDbSecretName" $))
*/}}
{{- define "datalab.postgres" -}}
{{- $root := index . 0 -}}
{{- $component := index . 1 -}}
{{- $db := index . 2 -}}
{{- $secret := index . 3 -}}
{{- $name := include "datalab.componentName" (list $root $component) -}}
{{- if $db.persistence.enabled }}
{{ include "datalab.pvc" (list $root $component $db.persistence) }}
---
{{- end }}
apiVersion: v1
kind: Service
metadata:
  name: {{ $name }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root $component) | nindent 4 }}
spec:
  type: ClusterIP
  selector:
    {{- include "datalab.componentSelectorLabels" (list $root $component) | nindent 4 }}
  ports:
    - name: postgres
      port: 5432
      targetPort: postgres
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $name }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root $component) | nindent 4 }}
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      {{- include "datalab.componentSelectorLabels" (list $root $component) | nindent 6 }}
  template:
    metadata:
      {{- include "datalab.podMetadata" (list $root $component $db) | nindent 6 }}
    spec:
      {{- include "datalab.podSpecCommon" (list $root $db) | nindent 6 }}
      automountServiceAccountToken: false
      enableServiceLinks: false
      containers:
        - name: postgres
          image: {{ include "datalab.image" (list $root $db.image) }}
          imagePullPolicy: {{ $db.image.pullPolicy }}
          securityContext:
            {{- include "datalab.containerSecurityContext" (list $root $db) | nindent 12 }}
          env:
            - name: POSTGRES_DB
              value: {{ $db.name | quote }}
            - name: POSTGRES_USER
              value: {{ $db.username | quote }}
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ $secret }}
                  key: {{ $db.auth.passwordKey }}
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          ports:
            - name: postgres
              containerPort: 5432
          startupProbe:
            exec:
              command: ["sh", "-c", "pg_isready -h 127.0.0.1 -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\""]
            periodSeconds: 2
            failureThreshold: 90
          readinessProbe:
            exec:
              command: ["sh", "-c", "pg_isready -h 127.0.0.1 -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\""]
            periodSeconds: 10
          livenessProbe:
            exec:
              command: ["sh", "-c", "pg_isready -h 127.0.0.1 -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\""]
            periodSeconds: 20
            failureThreshold: 6
          resources:
            {{- toYaml $db.resources | nindent 12 }}
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
            - name: run
              mountPath: /var/run/postgresql
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: data
          {{- include "datalab.volumeSource" (list $root $component $db.persistence) | nindent 10 }}
        - name: run
          emptyDir: {}
        - name: tmp
          emptyDir: {}
{{- end -}}

{{/* Init container that blocks until a Postgres accepts connections. Usage: include "datalab.waitForPostgres" (list $ "hms-db" .Values.hms.database) */}}
{{- define "datalab.waitForPostgres" -}}
{{- $root := index . 0 -}}
{{- $db := index . 2 -}}
- name: wait-for-db
  image: {{ include "datalab.image" (list $root $db.image) }}
  imagePullPolicy: {{ $db.image.pullPolicy }}
  securityContext:
    {{- include "datalab.containerSecurityContext" (list $root $db) | nindent 4 }}
  command:
    - sh
    - -c
    - |
      until pg_isready -h {{ include "datalab.componentName" (list $root (index . 1)) }} -p 5432 -U {{ $db.username }} -d {{ $db.name }}; do
        echo "waiting for {{ include "datalab.componentName" (list $root (index . 1)) }}"; sleep 2
      done
  resources:
    requests: {cpu: 10m, memory: 32Mi}
    limits: {memory: 64Mi}
{{- end -}}
