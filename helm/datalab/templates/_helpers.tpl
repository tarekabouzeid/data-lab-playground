{{/* Common labels (recommended app.kubernetes.io/* set) */}}
{{- define "datalab.labels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component | default .name }}
app.kubernetes.io/part-of: datalab
app.kubernetes.io/managed-by: {{ .ctx.Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .ctx.Chart.Name .ctx.Chart.Version | replace "+" "_" }}
{{- end }}

{{/* Selector labels: immutable, so only name + instance */}}
{{- define "datalab.selector" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
{{- end }}

{{/* repository:tag from an images.<x> entry */}}
{{- define "datalab.image" -}}
{{ .repository }}:{{ .tag }}
{{- end }}

{{/* Fail early, with a useful message, when a required API is missing from the cluster */}}
{{- define "datalab.requireAPI" -}}
{{- if and .ctx.Values.apiChecks (not (.ctx.Capabilities.APIVersions.Has .api)) -}}
{{- fail (printf "API %s is not available in this cluster. %s" .api .hint) -}}
{{- end -}}
{{- end }}

{{/* Pod-level defaults shared by all workloads. enableServiceLinks=false matters: a Service named
     "phoenix" would otherwise inject PHOENIX_PORT=tcp://..., which Phoenix parses as its own port setting. */}}
{{- define "datalab.podDefaults" -}}
enableServiceLinks: false
{{- end }}

{{/* Spark configuration (spark-defaults.conf mirror) shared by SparkConnect and SparkApplication.
     args: ctx + executor ({cores, memory}). The executor size is also a CR field, so it is forced into the conf
     here: two different values for the same property would make the winner depend on spark-submit argument order. */}}
{{- define "datalab.sparkConf" -}}
{{- $conf := deepCopy .ctx.Values.spark.sparkConf -}}
{{- $_ := set $conf "spark.executor.memory" (toString .executor.memory) -}}
{{- $_ := set $conf "spark.executor.cores" (toString .executor.cores) -}}
{{- toYaml $conf -}}
{{- end }}
