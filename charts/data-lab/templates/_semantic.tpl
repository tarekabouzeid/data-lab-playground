{{/*
Kubeflow Semantic Operator config files. Layout ported from the upstream chart
(kubeflow/semantic-operator@4fd7057, charts/semantic-operator/templates/configmap.yaml) rendered with
engine.type=trino; upstream defaults are kept for every key not set here (query bounds, HTTP timeouts, caches).
*/}}
{{- define "datalab.semanticEngine" -}}
{{- $root := index . 0 -}}
{{- dict "dialect" "trino" "connection" (dict "host" (include "datalab.host" (list $root "trino")) "port" 8080 "user" (index . 1)) | toYaml -}}
{{- end -}}

{{- define "datalab.semanticServerConfig" -}}
{{- $s := .Values.semantic.server -}}
{{- $auth := dict "mode" $s.auth.mode -}}
{{- if eq $s.auth.mode "jwt" -}}
{{- $j := $s.auth.jwt -}}
{{- $auth = dict "mode" "jwt" "jwksURL" $j.jwksURL "principalClaim" $j.principalClaim "roleClaim" $j.roleClaim -}}
{{- with $j.issuer }}{{ $_ := set $auth "issuer" . }}{{ end -}}
{{- with $j.audience }}{{ $_ := set $auth "audience" . }}{{ end -}}
{{- with $j.groupsClaim }}{{ $_ := set $auth "groupsClaim" . }}{{ end -}}
{{- end -}}
{{- $engine := include "datalab.semanticEngine" (list . $s.engineUser) | fromYaml -}}
{{- $_ := set $engine "identity" (dict "mode" "static") -}}
{{- $cfg := dict "logging" (dict "level" "info") "server" (dict "listenAddr" ":8090") "engine" $engine "auth" $auth "store" (dict "watchNamespace" .Release.Namespace) -}}
{{- mergeOverwrite $cfg (deepCopy $s.extraConfig) | toYaml -}}
{{- end -}}

{{- define "datalab.semanticManagerConfig" -}}
{{- $m := .Values.semantic.manager -}}
{{- $cfg := dict
  "leaderElection" (dict "leaderElect" true "resourceName" (include "datalab.componentName" (list . "semantic-manager")))
  "cache" (dict "watchNamespaces" (list .Release.Namespace))
  "engine" (include "datalab.semanticEngine" (list . $m.engineUser) | fromYaml)
  "controller" (dict "viewDatabase" "semantic_views" "resyncPeriod" $m.resyncPeriod) -}}
{{- toYaml $cfg -}}
{{- end -}}

{{/* One semantic Deployment (manager or server). Usage: include "datalab.semanticDeployment" (list $ "manager") */}}
{{- define "datalab.semanticDeployment" -}}
{{- $root := index . 0 -}}
{{- $role := index . 1 -}}
{{- $c := index $root.Values.semantic $role -}}
{{- $comp := printf "semantic-%s" $role -}}
{{- $cfg := ternary (include "datalab.semanticServerConfig" $root) (include "datalab.semanticManagerConfig" $root) (eq $role "server") -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "datalab.componentName" (list $root $comp) }}
  namespace: {{ $root.Release.Namespace }}
  labels:
    {{- include "datalab.componentLabels" (list $root $comp) | nindent 4 }}
spec:
  replicas: {{ ternary $c.replicas 1 (eq $role "server") }}
  selector:
    matchLabels:
      {{- include "datalab.componentSelectorLabels" (list $root $comp) | nindent 6 }}
  template:
    metadata:
      {{- include "datalab.podMetadata" (list $root $comp $c (dict "checksum/config" ($cfg | sha256sum))) | nindent 6 }}
    spec:
      {{- include "datalab.podSpecCommon" (list $root $c) | nindent 6 }}
      serviceAccountName: {{ include "datalab.componentName" (list $root $comp) }}
      enableServiceLinks: false
      containers:
        - name: {{ $role }}
          image: {{ include "datalab.image" (list $root $c.image) }}
          imagePullPolicy: {{ $c.image.pullPolicy }}
          args: ["--config=/etc/semantic-operator/config.yaml"]
          securityContext:
            {{- include "datalab.containerSecurityContext" (list $root $c) | nindent 12 }}
          {{- if eq $role "server" }}
          ports:
            - {name: http, containerPort: 8090}
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            initialDelaySeconds: 5
          readinessProbe:
            httpGet: {path: /readyz, port: http}
            initialDelaySeconds: 5
            periodSeconds: 10
          {{- else }}
          ports:
            - {name: metrics, containerPort: 8080}
            - {name: health, containerPort: 8081}
          livenessProbe:
            httpGet: {path: /healthz, port: health}
            initialDelaySeconds: 10
          readinessProbe:
            httpGet: {path: /readyz, port: health}
            initialDelaySeconds: 5
          {{- end }}
          resources:
            {{- toYaml $c.resources | nindent 12 }}
          volumeMounts:
            - name: config
              mountPath: /etc/semantic-operator
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: {{ include "datalab.componentName" (list $root $comp) }}
{{- end -}}


{{- /* One Ossie field. dim: "" (none) | "plain" | "time" */ -}}
{{- define "datalab.ossieField" -}}
- name: {{ index . 0 }}
  expression:
    dialects:
      - dialect: ANSI_SQL
        expression: {{ index . 1 }}
  description: {{ index . 2 | quote }}
  {{- if eq (index . 3) "time" }}
  dimension:
    is_time: true
  {{- else if eq (index . 3) "plain" }}
  dimension: {}
  {{- end -}}
{{- end -}}
