package edu.stanford.bmir.protege.web.client.issues;

import com.google.gwt.event.logical.shared.SelectionHandler;
import com.google.gwt.user.client.ui.IsWidget;
import edu.stanford.bmir.protege.web.client.pagination.HasPagination;
import edu.stanford.bmir.protege.web.shared.entity.CommentedEntityData;
import edu.stanford.bmir.protege.web.shared.entity.OWLEntityData;
import edu.stanford.bmir.protege.web.shared.issues.SortingKey;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 7 Mar 2017
 */
public interface CommentedEntitiesView extends IsWidget, HasPagination {

    interface GoToEntityHandler {
        void handleGoToEntity(@Nonnull OWLEntityData entityData);
    }

    void setEntities(@Nonnull List<CommentedEntityData> entities);

    void clear();

    void addEntity(@Nonnull CommentedEntityData entity);

    void removeEntity(@Nonnull CommentedEntityData entity);

    void refresh();

    SortingKey getSelectedSortingKey();

    void setSortingKeyChangedHandler(Runnable sortingKeyChangedHandler);

    void setSelectionHandler(@Nonnull SelectionHandler<CommentedEntityData> selectionHandler);

    void setPageCount(int pageCount);

    void setPageNumber(int pageNumber);

    int getPageNumber();

    void setPageNumberChangedHandler(HasPagination.PageNumberChangedHandler handler);

    void setGoToEntityHandler(@Nonnull GoToEntityHandler handler);
}
