import { useCallback, useContext, useEffect, useState } from "react";
import { Button, Columns } from "../../../components";
import { confirm, modal } from "../../../components/Modal/Modal";
import { Spinner } from "../../../components/Spinner/Spinner";
import { ApiContext } from "../../../providers/ApiProvider";
import { useProject } from "../../../providers/ProjectProvider";
import { StorageCard } from "./StorageCard";
import { ScrapingForm } from "./ScrapingForm";

export const ScrapingSet = ({ title, target, rootClass, buttonLabel }) => {
  const api = useContext(ApiContext);
  const { project } = useProject();
  const [storages, setStorages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [storageTypes, setStorageTypes] = useState([]);

  useEffect(() => {
    api
      .callApi("scrapingTypes", {
        params: {
          target,
        },
      })
      .then((types) => {
        setStorageTypes(types ?? []);
      });
  }, []);

  const fetchStorages = useCallback(async () => {
    if (!project.id) {
      console.warn("Project ID not provided");
      return;
    }

    setLoading(true);
    const result = await api.callApi("listStorages", {
      params: {
        project: project.id,
        target,
      },
    });

    const storageTypes = await api.callApi("scrapingTypes", {
      params: {
        target,
      },
    });

    setStorageTypes(storageTypes);

    if (result !== null) {
      setStorages(result);
      setLoaded(true);
    }

    setLoading(false);
  }, [project]);

  const showStorageFormModal = useCallback(
    (storage) => {
      const title = `Add Scraping Task`;

      const modalRef = modal({
        title,
        closeOnClickOutside: false,
        style: { width: 760 },
        body: (
          <ScrapingForm
            target={target}
            storage={storage}
            project={project.id}
            rootClass={rootClass}
            storageTypes={storageTypes}
            onSubmit={async () => {
              await fetchStorages();
              modalRef.close();
            }}
          />
        ),
      });
    },
    [project, fetchStorages, target, rootClass],
  );

  const onEditStorage = useCallback(
    async (storage) => {
      showStorageFormModal(storage);
    },
    [showStorageFormModal],
  );

  const onDeleteStorage = useCallback(
    async (storage) => {
      confirm({
        title: "Deleting storage",
        body: "This action cannot be undone. Are you sure?",
        buttonLook: "destructive",
        onOk: async () => {
          const response = await api.callApi("deleteStorage", {
            params: {
              type: storage.type,
              pk: storage.id,
              target,
            },
          });

          if (response !== null) fetchStorages();
        },
      });
    },
    [fetchStorages],
  );

  useEffect(() => {
    fetchStorages();
  }, [fetchStorages]);

  return (
    <Columns.Column title={title}>
      <div className={rootClass.elem("controls")}>
        <Button onClick={() => showStorageFormModal()}>{buttonLabel}</Button>
      </div>

      {loading && !loaded ? (
        <div className={rootClass.elem("empty")}>
          <Spinner size={32} />
        </div>
      ) : storages.length === 0 ? null : (
        storages.map((storage) => (
          <StorageCard
            key={storage.id}
            storage={storage}
            target={target}
            rootClass={rootClass}
            storageTypes={storageTypes}
            onEditStorage={onEditStorage}
            onDeleteStorage={onDeleteStorage}
          />
        ))
      )}
    </Columns.Column>
  );
};
