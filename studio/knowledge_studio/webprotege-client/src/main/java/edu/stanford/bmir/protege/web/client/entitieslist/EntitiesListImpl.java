package edu.stanford.bmir.protege.web.client.entitieslist;

import com.google.gwt.cell.client.AbstractCell;
import com.google.gwt.core.client.GWT;
import com.google.gwt.event.dom.client.MouseUpEvent;
import com.google.gwt.event.logical.shared.SelectionEvent;
import com.google.gwt.event.logical.shared.SelectionHandler;
import com.google.gwt.event.shared.HandlerRegistration;
import com.google.gwt.safehtml.shared.SafeHtmlBuilder;
import com.google.gwt.uibinder.client.UiBinder;
import com.google.gwt.uibinder.client.UiField;
import com.google.gwt.user.cellview.client.CellList;
import com.google.gwt.user.cellview.client.HasKeyboardSelectionPolicy;
import com.google.gwt.user.client.ui.Composite;
import com.google.gwt.user.client.ui.HTMLPanel;
import com.google.gwt.view.client.ListDataProvider;
import com.google.gwt.view.client.MultiSelectionModel;
import com.google.gwt.view.client.ProvidesKey;
import edu.stanford.bmir.protege.web.resources.WebProtegeCellListResources;
import edu.stanford.bmir.protege.web.shared.entity.OWLEntityData;
import org.semanticweb.owlapi.model.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

import static edu.stanford.bmir.protege.web.resources.WebProtegeClientBundle.BUNDLE;


/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 12/09/2013
 */
public class EntitiesListImpl<E extends OWLEntityData> extends Composite implements EntitiesList<E> {

    public static final OWLEntityVisitorEx<String> CSS_CLASS_NAME_VISITOR = new OWLEntityVisitorEx<String>() {
        @Nonnull
        @Override
        public String visit(@Nonnull OWLClass cls) {
            return BUNDLE.style().classIconInset();
        }

        @Nonnull
        @Override
        public String visit(@Nonnull OWLObjectProperty property) {
            return BUNDLE.style().objectPropertyIconInset();
        }

        @Nonnull
        @Override
        public String visit(@Nonnull OWLDataProperty property) {
            return BUNDLE.style().dataPropertyIconInset();
        }

        @Nonnull
        @Override
        public String visit(@Nonnull OWLNamedIndividual individual) {
            return BUNDLE.style().individualIconInset();
        }

        @Nonnull
        @Override
        public String visit(@Nonnull OWLDatatype datatype) {
            return BUNDLE.style().datatypeIconInset();
        }

        @Nonnull
        @Override
        public String visit(@Nonnull OWLAnnotationProperty property) {
            return BUNDLE.style().annotationPropertyIconInset();
        }
    };

    private ListDataProvider<E> listDataProvider = new ListDataProvider<E>();

    private EntitiesListItemRenderer<E> renderer = new DefaultEntitiesListItemRenderer<E>();

    private final MultiSelectionModel<E> selectionModel;

    interface EntitiesListImplUiBinder extends UiBinder<HTMLPanel, EntitiesListImpl<?>> {

    }

    private static EntitiesListImplUiBinder ourUiBinder = GWT.create(EntitiesListImplUiBinder.class);

    @UiField(provided = true)
    protected CellList<E> cellList;

    public EntitiesListImpl() {
        OWLEntityDataKeyProvider keyProvider = new OWLEntityDataKeyProvider();
        cellList = new CellList<E>(new OWLEntityDataCell(), WebProtegeCellListResources.INSTANCE, keyProvider);
        listDataProvider.addDataDisplay(cellList);
        cellList.setKeyboardSelectionPolicy(HasKeyboardSelectionPolicy.KeyboardSelectionPolicy.ENABLED);
        selectionModel = new MultiSelectionModel<E>(keyProvider);
        cellList.setSelectionModel(selectionModel);
        selectionModel.addSelectionChangeHandler(event -> SelectionEvent.fire(EntitiesListImpl.this, getSingleSelection()));
        cellList.addDomHandler(mouseUpEvent -> SelectionEvent.fire(EntitiesListImpl.this, getSingleSelection()), MouseUpEvent.getType());
        HTMLPanel rootElement = ourUiBinder.createAndBindUi(this);
        initWidget(rootElement);
    }

    @Nullable
    private E getSingleSelection() {
        Set<E> sel = selectionModel.getSelectedSet();
        if(sel.isEmpty()) {
            return null;
        }
        else if(sel.size() == 1) {
            return sel.iterator().next();
        }
        else {
            return sel.iterator().next();
        }
    }


    @Override
    public void remove(E entity) {
        listDataProvider.getList().remove(entity);
    }

    @Override
    public void removeAll(Collection<E> entities) {
        listDataProvider.getList().removeAll(entities);
    }

    @Override
    public void addAll(Collection<E> entities) {
        List<E> existing = listDataProvider.getList();
        List<E> sortedList = new ArrayList<E>(entities);
        Collections.sort(sortedList, (o1, o2) -> o1.getBrowserText().compareToIgnoreCase(o2.getBrowserText()));
        existing.addAll(0, sortedList);
        cellList.setPageSize(existing.size());
    }

    @Override
    public void clear() {
        listDataProvider.getList().clear();
    }

    @Override
    public void setListData(List<E> entities) {
        listDataProvider.getList().clear();
        listDataProvider.getList().addAll(entities);
        cellList.setPageSize(listDataProvider.getList().size());
    }

    @Override
    public void setSelectedEntity(E selectedEntity) {
        selectionModel.clear();
        selectionModel.setSelected(selectedEntity, true);
    }

    @Override
    public Optional<E> getSelectedEntity() {
        E sel = getSingleSelection();
        return Optional.ofNullable(sel);
    }

    @Override
    public Set<E> getSelectedEntities() {
        return selectionModel.getSelectedSet();
    }

    @Override
    public HandlerRegistration addSelectionHandler(SelectionHandler<E> handler) {
        return addHandler(handler, SelectionEvent.getType());
    }



    private class OWLEntityDataCell extends AbstractCell<E> {

        @Override
        public void render(Context context, E value, SafeHtmlBuilder safeHtmlBuilder) {
            StringBuilder sb = new StringBuilder();
            String cssClassName = value.accept(CSS_CLASS_NAME_VISITOR, BUNDLE.style().emptyIconInset());
            sb.append("<div class=\"").append(cssClassName).append("\" style=\"line-height: 20px;\">");
            renderer.render(value, sb);
            sb.append("</div>");
            safeHtmlBuilder.appendHtmlConstant(sb.toString());

        }
    }

    private class OWLEntityDataKeyProvider implements ProvidesKey<E> {

        @Override
        public Object getKey(E item) {
            return item.getEntity();
        }
    }
}



