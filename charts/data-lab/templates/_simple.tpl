{{/*
Deployment + Service + optional PVC for a single-container component.
Usage: include "datalab.simpleComponent" (dict "root" $ "component" "qdrant" "values" .Values.qdrant
         "ports" (list (dict "name" "http" "port" 6333)) "container" "<container YAML without name/image/securityContext/ports/resources>"
         "dataPath" "/qdrant/storage")
*/}}
{{- define "datalab.simpleComponent" -}}
{{- $root := .root -}}
{{- $c := .values -}}
{{- $name := include "datalab.componentName" (list $root .component) -}}
{{- if and $c.persistence.enabled (not $c.persistence.existingClaim) }}
{{ include "datalab.pvc" (list $root .component $c.persistence) }}
---
{{- end }}
apiVersion: v1
kind: Service
metadata:
  name: {{ $name }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root .component) | nindent 4 }}
spec:
  type: ClusterIP
  selector:
    {{- include "datalab.componentSelectorLabels" (list $root .component) | nindent 4 }}
  ports:
    {{- range .ports }}
    - name: {{ .name }}
      port: {{ .port }}
      targetPort: {{ .name }}
    {{- end }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $name }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root .component) | nindent 4 }}
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      {{- include "datalab.componentSelectorLabels" (list $root .component) | nindent 6 }}
  template:
    metadata:
      {{- include "datalab.podMetadata" (list $root .component $c) | nindent 6 }}
    spec:
      {{- include "datalab.podSpecCommon" (list $root $c) | nindent 6 }}
      automountServiceAccountToken: false
      enableServiceLinks: false
      {{- with .podExtra }}
      {{- . | nindent 6 }}
      {{- end }}
      containers:
        - name: {{ .component }}
          image: {{ include "datalab.image" (list $root $c.image) }}
          imagePullPolicy: {{ $c.image.pullPolicy }}
          securityContext:
            {{- include "datalab.containerSecurityContext" (list $root $c) | nindent 12 }}
          ports:
            {{- range .ports }}
            - name: {{ .name }}
              containerPort: {{ .port }}
            {{- end }}
          {{- .container | nindent 10 }}
          {{- with .extraResources }}
          resources:
            {{- toYaml (mergeOverwrite (deepCopy $c.resources) .) | nindent 12 }}
          {{- else }}
          resources:
            {{- toYaml $c.resources | nindent 12 }}
          {{- end }}
          volumeMounts:
            - name: data
              mountPath: {{ .dataPath }}
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: data
          {{- include "datalab.volumeSource" (list $root .component $c.persistence) | nindent 10 }}
        - name: tmp
          emptyDir: {}
{{- end -}}
