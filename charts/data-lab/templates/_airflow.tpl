{{/* Env shared by every Airflow container. Usage: include "datalab.airflowEnv" (list $ <dags folder>) */}}
{{- define "datalab.airflowEnv" -}}
{{- $root := index . 0 -}}
{{- $c := $root.Values.airflow -}}
{{- $secrets := include "datalab.airflowSecretsSecretName" $root -}}
- name: AIRFLOW_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "datalab.airflowDbSecretName" $root }}
      key: {{ $c.database.auth.passwordKey }}
- name: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
  value: "postgresql+psycopg2://{{ $c.database.username }}:$(AIRFLOW_DB_PASSWORD)@{{ include "datalab.componentName" (list $root "airflow-db") }}:5432/{{ $c.database.name }}"
- name: AIRFLOW__CORE__FERNET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ $secrets }}
      key: fernet-key
- name: AIRFLOW__API_AUTH__JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ $secrets }}
      key: jwt-secret
- name: AIRFLOW__API__SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ $secrets }}
      key: api-secret-key
- name: AIRFLOW__CORE__EXECUTOR
  value: LocalExecutor
- name: AIRFLOW__CORE__LOAD_EXAMPLES
  value: "False"
- name: AIRFLOW__CORE__DAGS_FOLDER
  value: {{ index . 1 | quote }}
- name: AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL
  value: "30"
- name: AIRFLOW__CORE__AUTH_MANAGER
  value: airflow.api_fastapi.auth.managers.simple.simple_auth_manager.SimpleAuthManager
- name: AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS
  value: "{{ $c.auth.username }}:admin{{ if $root.Values.airflowMcp.enabled }},{{ $root.Values.airflowMcp.username }}:{{ include "datalab.airflowMcpRole" $root }}{{ end }}"
- name: AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE
  value: /opt/airflow/auth/passwords.json
- name: AIRFLOW__CORE__EXECUTION_API_SERVER_URL
  value: http://localhost:8080/execution/
- name: AIRFLOW__LOGGING__BASE_LOG_FOLDER
  value: /opt/airflow/logs
- name: CONNECTION_CHECK_MAX_COUNT
  value: "0"
{{- if $c.labPipelines.enabled }}
{{- $v := $root.Values }}
{{- if $v.spark.enabled }}
{{- $sc := include "datalab.host" (list $root "spark-connect") }}
{{- if $v.spark.auth.enabled }}
- name: SPARK_CONNECT_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "datalab.sparkSecretName" $root }}
      key: {{ $v.spark.auth.tokenKey }}
- name: SPARK_REMOTE
  value: "sc://{{ $sc }}:15002/;authorization=Bearer $(SPARK_CONNECT_TOKEN)"
{{- else }}
- name: SPARK_REMOTE
  value: "sc://{{ $sc }}:15002"
{{- end }}
{{- end }}
{{- if $v.trino.enabled }}
- name: TRINO_HOST
  value: {{ include "datalab.host" (list $root "trino") | quote }}
- name: TRINO_PORT
  value: "8080"
{{- end }}
{{- if and $v.semantic.enabled $v.semantic.models.dbtDemo.enabled }}
- name: SEMANTIC_MODEL_RESOURCE
  value: {{ include "datalab.componentName" (list $root "dbt-demo") | quote }}
{{- end }}
{{- end }}
{{- with $c.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "datalab.airflowMounts" -}}
- name: dags
  mountPath: {{ index . 1 }}
  readOnly: true
- name: logs
  mountPath: /opt/airflow/logs
- name: tmp
  mountPath: /tmp
{{- end }}

{{/* git-sync container body. Usage: include "datalab.gitSyncContainer" (list $ <oneTime bool>) */}}
{{- define "datalab.gitSyncContainer" -}}
{{- $root := index . 0 -}}
{{- $c := $root.Values.airflow -}}
{{- $g := $c.dags.gitSync -}}
image: {{ include "datalab.image" (list $root $g.image) }}
imagePullPolicy: {{ $g.image.pullPolicy }}
securityContext:
  {{- include "datalab.containerSecurityContext" (list $root $c) | nindent 2 }}
env:
  - name: GITSYNC_REPO
    value: {{ $g.repo | quote }}
  - name: GITSYNC_REF
    value: {{ $g.ref | quote }}
  - name: GITSYNC_ROOT
    value: /git
  - name: GITSYNC_LINK
    value: repo
  - name: GITSYNC_DEPTH
    value: {{ $g.depth | quote }}
  - name: GITSYNC_PERIOD
    value: {{ $g.period | quote }}
  - name: GITSYNC_ONE_TIME
    value: {{ index . 1 | quote }}
  - name: HOME
    value: /tmp
  {{- if $g.existingSecret }}
  - name: GITSYNC_USERNAME
    valueFrom:
      secretKeyRef:
        name: {{ $g.existingSecret }}
        key: username
        optional: true
  - name: GITSYNC_PASSWORD
    valueFrom:
      secretKeyRef:
        name: {{ $g.existingSecret }}
        key: password
        optional: true
  {{- if or (hasPrefix "git@" $g.repo) (hasPrefix "ssh://" $g.repo) }}
  - name: GITSYNC_SSH_KEY_FILE
    value: /etc/git-secret/ssh-privatekey
  - name: GITSYNC_SSH_KNOWN_HOSTS_FILE
    value: /etc/git-secret/known_hosts
  {{- end }}
  {{- end }}
resources:
  {{- toYaml $g.resources | nindent 2 }}
volumeMounts:
  - name: dags
    mountPath: /git
  - name: tmp
    mountPath: /tmp
  {{- if $g.existingSecret }}
  - name: git-secret
    mountPath: /etc/git-secret
    readOnly: true
  {{- end }}
{{- end -}}
