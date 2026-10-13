{{/* ---------- names ---------- */}}

{{- define "datalab.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Release prefix for every resource. Kept to 40 chars so "<fullname>-<component>" stays a
valid DNS label (63) and Spark executor pod names fit.
*/}}
{{- define "datalab.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 40 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 40 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* "<fullname>-<component>". Usage: include "datalab.componentName" (list $ "hms") */}}
{{- define "datalab.componentName" -}}
{{- $root := index . 0 -}}
{{- printf "%s-%s" (include "datalab.fullname" $root) (index . 1) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "datalab.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* In-cluster FQDN of a component service. Usage: include "datalab.host" (list $ "hms") */}}
{{- define "datalab.host" -}}
{{- $root := index . 0 -}}
{{- printf "%s.%s.svc" (include "datalab.componentName" .) $root.Release.Namespace -}}
{{- end -}}

{{/* ---------- labels ---------- */}}

{{- define "datalab.selectorLabels" -}}
app.kubernetes.io/name: {{ include "datalab.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "datalab.labels" -}}
helm.sh/chart: {{ include "datalab.chart" . }}
{{ include "datalab.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: data-lab
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/* Usage: include "datalab.componentLabels" (list $ "hms") */}}
{{- define "datalab.componentLabels" -}}
{{ include "datalab.labels" (index . 0) }}
app.kubernetes.io/component: {{ index . 1 }}
{{- end -}}

{{/* Usage: include "datalab.componentSelectorLabels" (list $ "hms") */}}
{{- define "datalab.componentSelectorLabels" -}}
{{ include "datalab.selectorLabels" (index . 0) }}
app.kubernetes.io/component: {{ index . 1 }}
{{- end -}}

{{/* ---------- images ---------- */}}

{{/* Usage: include "datalab.image" (list $ .Values.hms.image) */}}
{{- define "datalab.image" -}}
{{- $root := index . 0 -}}
{{- $img := index . 1 -}}
{{- $registry := $root.Values.global.imageRegistry | default $img.registry -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $img.repository ($img.tag | toString) -}}
{{- else -}}
{{- printf "%s:%s" $img.repository ($img.tag | toString) -}}
{{- end -}}
{{- end -}}

{{/* ---------- pod spec fragments ---------- */}}

{{/* Pod metadata labels + annotations. Usage: include "datalab.podMetadata" (list $ "hms" .Values.hms (dict "checksum/config" "...")) */}}
{{- define "datalab.podMetadata" -}}
{{- $root := index . 0 -}}
{{- $comp := index . 2 -}}
{{- $extraAnn := dict -}}
{{- if gt (len .) 3 }}{{ $extraAnn = index . 3 }}{{ end -}}
labels:
  {{- include "datalab.componentLabels" (list $root (index . 1)) | nindent 2 }}
  {{- with (merge (dict) ($comp.podLabels | default dict) $root.Values.global.podLabels) }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
{{- $ann := merge (dict) $extraAnn ($comp.podAnnotations | default dict) $root.Values.global.podAnnotations }}
{{- with $ann }}
annotations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
Scheduling + security fields shared by every pod.
Usage: include "datalab.podSpecCommon" (list $ .Values.hms)
*/}}
{{- define "datalab.podSpecCommon" -}}
{{- $root := index . 0 -}}
{{- $comp := index . 1 -}}
{{- with $root.Values.global.imagePullSecrets }}
imagePullSecrets:
  {{- range . }}
  - name: {{ . }}
  {{- end }}
{{- end }}
{{- with ($comp.runtimeClassName | default $root.Values.global.runtimeClassName) }}
runtimeClassName: {{ . }}
{{- end }}
{{- with ($comp.nodeSelector | default $root.Values.global.nodeSelector) }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with ($comp.tolerations | default $root.Values.global.tolerations) }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $comp.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
securityContext:
  {{- toYaml $comp.podSecurityContext | nindent 2 }}
{{- end -}}

{{/* Container securityContext. Usage: include "datalab.containerSecurityContext" (list $ .Values.hms) */}}
{{- define "datalab.containerSecurityContext" -}}
{{- $root := index . 0 -}}
{{- $comp := index . 1 -}}
{{- toYaml (mergeOverwrite (deepCopy $root.Values.global.containerSecurityContext) ($comp.securityContext | default dict)) -}}
{{- end -}}

{{/* ---------- storage ---------- */}}

{{/* storageClassName line. Usage: include "datalab.storageClass" (list $ .Values.hms.database.persistence) */}}
{{- define "datalab.storageClass" -}}
{{- $root := index . 0 -}}
{{- $sc := (index . 1).storageClass | default $root.Values.global.storageClass -}}
{{- if eq $sc "-" }}
storageClassName: ""
{{- else if $sc }}
storageClassName: {{ $sc }}
{{- end }}
{{- end -}}

{{/* PVC manifest. Usage: include "datalab.pvc" (list $ "hms-db" .Values.hms.database.persistence) */}}
{{- define "datalab.pvc" -}}
{{- $root := index . 0 -}}
{{- $p := index . 2 -}}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "datalab.componentName" (list $root (index . 1)) }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root (index . 1)) | nindent 4 }}
spec:
  accessModes:
    {{- toYaml $p.accessModes | nindent 4 }}
  {{- include "datalab.storageClass" (list $root $p) | nindent 2 }}
  resources:
    requests:
      storage: {{ $p.size }}
{{- end -}}

{{/* Volume source: PVC or emptyDir. Usage: include "datalab.volumeSource" (list $ "hms-db" .Values.hms.database.persistence) */}}
{{- define "datalab.volumeSource" -}}
{{- if (index . 2).enabled }}
persistentVolumeClaim:
  claimName: {{ include "datalab.componentName" (list (index . 0) (index . 1)) }}
{{- else }}
emptyDir: {}
{{- end }}
{{- end -}}

{{/* ---------- secrets ---------- */}}

{{/*
Base64 value for a generated Secret key: reuse the live value (stable across upgrades),
otherwise random. lookup is empty under `helm template` / client dry-run (see README).
Usage: include "datalab.secretValue" (list $ "<secret name>" "<key>" 32)
*/}}
{{- define "datalab.secretValue" -}}
{{- $root := index . 0 -}}
{{- $existing := lookup "v1" "Secret" $root.Release.Namespace (index . 1) -}}
{{- if and $existing $existing.data (hasKey $existing.data (index . 2)) -}}
{{- index $existing.data (index . 2) -}}
{{- else -}}
{{- randAlphaNum (index . 3) | b64enc -}}
{{- end -}}
{{- end -}}

{{- define "datalab.s3SecretName" -}}
{{- .Values.s3.auth.existingSecret | default (include "datalab.componentName" (list . "s3")) -}}
{{- end -}}

{{- define "datalab.hmsDbSecretName" -}}
{{- .Values.hms.database.auth.existingSecret | default (include "datalab.componentName" (list . "hms-db")) -}}
{{- end -}}

{{- define "datalab.sparkSecretName" -}}
{{- .Values.spark.auth.existingSecret | default (include "datalab.componentName" (list . "spark-connect")) -}}
{{- end -}}

{{- define "datalab.airflowAuthSecretName" -}}
{{- .Values.airflow.auth.existingSecret | default (include "datalab.componentName" (list . "airflow")) -}}
{{- end -}}

{{- define "datalab.airflowSecretsSecretName" -}}
{{- .Values.airflow.secrets.existingSecret | default (include "datalab.componentName" (list . "airflow")) -}}
{{- end -}}

{{- define "datalab.airflowDbSecretName" -}}
{{- .Values.airflow.database.auth.existingSecret | default (include "datalab.componentName" (list . "airflow-db")) -}}
{{- end -}}

{{- define "datalab.jupyterSecretName" -}}
{{- .Values.jupyter.auth.existingSecret | default (include "datalab.componentName" (list . "jupyter")) -}}
{{- end -}}

{{- define "datalab.qdrantSecretName" -}}
{{- .Values.qdrant.auth.existingSecret | default (include "datalab.componentName" (list . "qdrant")) -}}
{{- end -}}

{{- define "datalab.phoenixSecretName" -}}
{{- .Values.phoenix.auth.existingSecret | default (include "datalab.componentName" (list . "phoenix")) -}}
{{- end -}}

{{/* ---------- endpoints ---------- */}}

{{- define "datalab.s3Endpoint" -}}
{{- .Values.s3.endpoint | default (printf "http://%s:8333" (include "datalab.host" (list . "seaweedfs"))) -}}
{{- end -}}

{{- define "datalab.hmsThriftUri" -}}
{{- printf "thrift://%s:9083" (include "datalab.host" (list . "hms")) -}}
{{- end -}}

{{- define "datalab.icebergRestUri" -}}
{{- printf "http://%s:9084/iceberg" (include "datalab.host" (list . "hms")) -}}
{{- end -}}

{{/* S3 credential env vars (S3_ACCESS_KEY / S3_SECRET_KEY). Usage: include "datalab.s3Env" . */}}
{{- define "datalab.s3Env" -}}
- name: S3_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "datalab.s3SecretName" . }}
      key: {{ .Values.s3.auth.accessKeyKey }}
- name: S3_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "datalab.s3SecretName" . }}
      key: {{ .Values.s3.auth.secretKeyKey }}
{{- end -}}

{{/* ---------- Spark event logs ---------- */}}

{{- define "datalab.eventLogDir" -}}
{{- .Values.sparkHistory.eventLogDir | default (printf "s3a://%s/spark-events" .Values.s3.bucket) -}}
{{- end -}}

{{/*
Init container that creates the Spark event log "directory" in S3 (waits for S3 first). Spark refuses to start
when spark.eventLog.dir does not exist, and the History Server needs it too. Uses the Hadoop client bundled in
the Spark image: the generated core-site.xml keeps the S3 keys out of the process arguments.
Usage: include "datalab.eventLogDirInit" .
*/}}
{{- define "datalab.eventLogDirInit" -}}
- name: event-log-dir
  image: {{ include "datalab.image" (list . .Values.spark.image) }}
  imagePullPolicy: {{ .Values.spark.image.pullPolicy }}
  securityContext:
    {{- include "datalab.containerSecurityContext" (list . .Values.spark) | nindent 4 }}
  command:
    - bash
    - -c
    - |
      mkdir -p /tmp/hconf
      cat > /tmp/hconf/core-site.xml <<XML
      <configuration>
        <property><name>fs.s3a.endpoint</name><value>{{ include "datalab.s3Endpoint" . }}</value></property>
        <property><name>fs.s3a.endpoint.region</name><value>{{ .Values.s3.region }}</value></property>
        <property><name>fs.s3a.path.style.access</name><value>{{ .Values.s3.pathStyleAccess }}</value></property>
        <property><name>fs.s3a.connection.ssl.enabled</name><value>{{ hasPrefix "https://" (include "datalab.s3Endpoint" .) }}</value></property>
        <property><name>fs.s3a.access.key</name><value>${S3_ACCESS_KEY}</value></property>
        <property><name>fs.s3a.secret.key</name><value>${S3_SECRET_KEY}</value></property>
        <property><name>fs.s3a.aws.credentials.provider</name><value>org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider</value></property>
      </configuration>
      XML
      until java -cp "/tmp/hconf:/opt/spark/jars/*" org.apache.hadoop.fs.FsShell -mkdir -p "{{ include "datalab.eventLogDir" . }}" 2>/tmp/hconf/err; do
        echo "waiting for {{ include "datalab.eventLogDir" . }}: $(grep -m1 -oE '[A-Za-z.]+Exception[^:]*: .{0,150}' /tmp/hconf/err)"
        sleep 3
      done
  env:
    {{- include "datalab.s3Env" . | nindent 4 }}
  resources:
    requests: {cpu: 50m, memory: 256Mi}
    limits: {memory: 512Mi}
  volumeMounts:
    - name: tmp
      mountPath: /tmp
{{- end -}}
