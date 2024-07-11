package edu.stanford.bmir.protege.web.server.project;

import org.semanticweb.owlapi.io.OWLOntologyDocumentSource;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyCreationException;
import org.semanticweb.owlapi.model.OWLOntologyLoaderConfiguration;
import org.semanticweb.owlapi.model.OWLOntologyManager;

/**
 * @author Matthew Horridge,
 *         Stanford University,
 *         Bio-Medical Informatics Research Group
 *         Date: 19/02/2014
 */
public class RawProjectSourcesImporter {

    private OWLOntologyManager manager;

    private OWLOntologyLoaderConfiguration loaderConfig;

    public RawProjectSourcesImporter(OWLOntologyManager manager, OWLOntologyLoaderConfiguration loaderConfig) {
        this.manager = manager;
        this.loaderConfig = loaderConfig;
    }

    public OWLOntology importRawProjectSources(RawProjectSources projectSources) throws OWLOntologyCreationException {
        try {
            manager.getIRIMappers().add(projectSources.getOntologyIRIMapper());
            OWLOntology ontology = null;
            for (OWLOntologyDocumentSource documentSource : projectSources.getDocumentSources()) {
                ontology = manager.loadOntologyFromOntologyDocument(documentSource, loaderConfig);
            }
            return ontology;
        } finally {
            manager.getIRIMappers().remove(projectSources.getOntologyIRIMapper());
        }
    }

}
