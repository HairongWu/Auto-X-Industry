package edu.stanford.bmir.protege.web.client.editor;

import edu.stanford.bmir.protege.web.client.lang.DisplayNameRenderer;
import edu.stanford.bmir.protege.web.client.portlet.AbstractWebProtegePortletPresenter;
import edu.stanford.bmir.protege.web.client.portlet.PortletUi;
import edu.stanford.bmir.protege.web.shared.event.WebProtegeEventBus;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.client.selection.SelectionModel;
import edu.stanford.webprotege.shared.annotations.Portlet;

import javax.annotation.Nonnull;
import javax.inject.Inject;

import static com.google.common.base.Preconditions.checkNotNull;
import static org.semanticweb.owlapi.model.EntityType.NAMED_INDIVIDUAL;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 22 Mar 2017
 */
@Portlet(
        id = "portlets.IndividualEditor",
        title = "Individual",
        tooltip = "Provides an editor that allows individual descriptions to be edited"
)
public class NamedIndividualEditorPortletPresenter extends AbstractWebProtegePortletPresenter {

    private final EditorPortletPresenter editorPresenter;

    @Inject
    public NamedIndividualEditorPortletPresenter(@Nonnull SelectionModel selectionModel,
                                                 @Nonnull ProjectId projectId,
                                                 @Nonnull EditorPortletPresenter editorPresenter, DisplayNameRenderer displayNameRenderer) {
        super(selectionModel, projectId, displayNameRenderer);
        this.editorPresenter = checkNotNull(editorPresenter);
    }

    @Override
    public void startPortlet(PortletUi portletUi, WebProtegeEventBus eventBus) {
        editorPresenter.setDisplayedTypes(NAMED_INDIVIDUAL);
        editorPresenter.start(portletUi, eventBus);
    }
}
