import React from "react";
import { useHistory } from "react-router";
import { Button, ToggleItems } from "../../components";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { cn } from "../../utils/bem";
import "./CreateProject.styl";

import { useDraftProject } from "../CreateProject/utils/useDraftProject";

const ProjectName = ({ name, setName, onSaveName, onSubmit, error, description, setDescription, show = true }) =>
  !show ? null : (
    <form
      className={cn("project-name")}
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <div className="field field--wide">
        <label htmlFor="project_name">Server Name</label>
        <input
          name="name"
          id="project_name"
          value="Server Name"
          onChange={(e) => setName(e.target.value)}
          onBlur={onSaveName}
        />
        {error && <span className="error">{error}</span>}
      </div>
      <div className="field field--wide">
        <label htmlFor="project_description">Backend URL</label>
        <input
          name="description"
          id="project_description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onBlur={onSaveName}
        />
      </div>
      
    </form>
  );

export const CreateProject = ({ onClose }) => {
  const [step, setStep] = React.useState("name"); // name | import | config
  const [waiting, setWaitingStatus] = React.useState(false);

  const project = useDraftProject();
  const history = useHistory();
  const api = useAPI();

  const [name, setName] = React.useState("");
  const [error, setError] = React.useState();
  const [description, setDescription] = React.useState("");
  const [config, setConfig] = React.useState("<View></View>");

  React.useEffect(() => {
    setError(null);
  }, [name]);

  const rootClass = cn("create-project");

  // name intentionally skipped from deps:
  // this should trigger only once when we got project loaded
  React.useEffect(() => project && !name && setName(project.title), [project]);

  const projectBody = React.useMemo(
    () => ({
      title: name,
      description,
      label_config: config,
    }),
    [name, description, config],
  );

  const onCreate = React.useCallback(async () => {
    setWaitingStatus(true);
    const response = await api.callApi("projectGenBackend", {
      params: {
        pk: project.id,
      },
      body: projectBody,
    });

    setWaitingStatus(false);

    if (response !== null) {
      history.push(`/projects/${response.id}/data`);
    }
  }, [project, projectBody]);

  const onSaveName = async () => {
    if (error) return;
    const res = await api.callApi("updateProjectRaw", {
      params: {
        pk: project.id,
      },
      body: {
        title: name,
      },
    });

    if (res.ok) return;
    const err = await res.json();

    setError(err.validation_errors?.title);
  };

  const onDelete = React.useCallback(async () => {
    setWaitingStatus(true);
    if (project)
      await api.callApi("deleteProject", {
        params: {
          pk: project.id,
        },
      });
    setWaitingStatus(false);
    history.replace("/projects");
    onClose?.();
  }, [project]);

  return (
    <Modal onHide={onDelete} closeOnClickOutside={false} allowToInterceptEscape fullscreen visible bare>
      <div className={rootClass}>
        <Modal.Header>
          <Space>
            <Button look="danger" size="compact" onClick={onDelete} waiting={waiting}>
              Cancel
            </Button>
            <Button
              look="primary"
              size="compact"
              onClick={onCreate}
              waiting={waiting}
              disabled={!project || error}
            >
              Connect to Auto-X Server
            </Button>
          </Space>
        </Modal.Header>
        <ProjectName
          name={name}
          setName={setName}
          error={error}
          onSaveName={onSaveName}
          onSubmit={onCreate}
          description={description}
          setDescription={setDescription}
          show={step === "name"}
        />
      </div>
    </Modal>
  );
};
