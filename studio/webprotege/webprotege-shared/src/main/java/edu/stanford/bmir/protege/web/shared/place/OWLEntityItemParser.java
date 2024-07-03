package edu.stanford.bmir.protege.web.shared.place;

import org.semanticweb.owlapi.model.*;

import java.util.Optional;

/**
 * @author Matthew Horridge, Stanford University, Bio-Medical Informatics Research Group, Date: 19/05/2014
 */
public abstract class OWLEntityItemParser<E extends OWLEntity> implements ItemTokenParser.ItemParser {

    public static final String IRI_START = "<";

    public static final String IRI_END = ">";

    private EntityType<E> entityType;

    private PrefixManager pm;

    private OWLDataFactory dataFactory;

    public OWLEntityItemParser(EntityType<E> entityType, OWLDataFactory dataFactory, PrefixManager pm) {
        this.entityType = entityType;
        this.dataFactory = dataFactory;
        this.pm = pm;
    }

    public Optional<Item> parseItem(String content) {
        final Optional<IRI> iri = parseIRI(content);
        if(!iri.isPresent()) {
            return Optional.empty();
        }
        E entity = dataFactory.getOWLEntity(entityType, iri.get());
        return Optional.of(createItem(entity));
    }

    private Optional<IRI> parseIRI(String iri) {
        Optional<IRI> result;
        if (iri.startsWith(IRI_START) && iri.endsWith(IRI_END)) {
            result = getIRIFromQuotedIRI(iri);
        } else {
            result = getIRIFromPrefixIRI(iri);
        }
        return result;
    }

    private Optional<IRI> getIRIFromPrefixIRI(String content) {
        int colonIndex = content.indexOf(":");
        if (colonIndex == -1) {
            return Optional.empty();
        }
        String prefixName = content.substring(0, colonIndex + 1);
        if (!pm.containsPrefixMapping(prefixName)) {
            return Optional.empty();
        }
        IRI iri = pm.getIRI(content);
        return Optional.of(iri);
    }

    private Optional<IRI> getIRIFromQuotedIRI(String quotedIRI) {
        return Optional.of(IRI.create(quotedIRI.substring(1, quotedIRI.length() - 1)));
    }

    protected abstract Item<E> createItem(E entity);
}
