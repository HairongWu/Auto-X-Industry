/*
 * Copyright 2021, OpenRemote Inc.
 *
 * See the CONTRIBUTORS.txt file in the distribution for a
 * full listing of individual contributors.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package org.openremote.model.custom;

import org.openremote.model.asset.Asset;
import org.openremote.model.asset.AssetDescriptor;
import org.openremote.model.asset.impl.GasAsset;
import org.openremote.model.asset.impl.GasSupplierAsset;
import org.openremote.model.value.AttributeDescriptor;
import org.openremote.model.value.ValueDescriptor;

import jakarta.persistence.Entity;
import org.openremote.model.value.ValueType;

import java.util.Optional;

import static org.openremote.model.Constants.UNITS_KILO;
import static org.openremote.model.Constants.UNITS_WATT;

/**
 * This is an example of a custom {@link Asset} type; this must be registered via an
 * {@link org.openremote.model.AssetModelProvider} and must conform to the following requirements:
 *
 * <ul>
 * <li>Must have {@link Entity} annotation
 * <li>Optionally add {@link org.openremote.model.value.ValueDescriptor}s
 * <li>Optionally add {@link org.openremote.model.value.MetaItemDescriptor}s
 * <li>Optionally add {@link org.openremote.model.value.AttributeDescriptor}s
 * <li>Must have a public static final {@link org.openremote.model.asset.AssetDescriptor}
 * <li>Must have a protected no args constructor (for hydrators i.e. JPA/Jackson)
 * <li>For a given {@link Asset} type only one {@link org.openremote.model.asset.AssetDescriptor} can exist
 * <li>{@link org.openremote.model.value.AttributeDescriptor}s that override a super class descriptor cannot change the
 * value type; just the formatting etc.
 * <li>{@link org.openremote.model.value.MetaItemDescriptor}s names must be unique
 * <li>{@link org.openremote.model.value.ValueDescriptor}s names must be unique
 * </ul>
 */
@Entity
public class GasConsumerAsset extends Asset<GasConsumerAsset> {


    public static final AssetDescriptor<GasConsumerAsset> DESCRIPTOR = new AssetDescriptor<>("power-plug", "8A293D", GasConsumerAsset.class);

    public static final AttributeDescriptor<Double> POWER_SETPOINT = GasAsset.POWER_SETPOINT.withOptional(true);
    public static final AttributeDescriptor<Double> POWER_IMPORT_MIN = GasAsset.POWER_IMPORT_MIN.withOptional(true);
    public static final AttributeDescriptor<Double> POWER_IMPORT_MAX = GasAsset.POWER_IMPORT_MAX.withOptional(true);
    public static final AttributeDescriptor<Double> POWER_EXPORT_MIN = GasAsset.POWER_EXPORT_MIN.withOptional(true);
    public static final AttributeDescriptor<Double> POWER_EXPORT_MAX = GasAsset.POWER_EXPORT_MAX.withOptional(true);
    public static final AttributeDescriptor<Double> ENERGY_EXPORT_TOTAL = GasAsset.ENERGY_EXPORT_TOTAL.withOptional(true);
    public static final AttributeDescriptor<Integer> EFFICIENCY_IMPORT = GasAsset.EFFICIENCY_IMPORT.withOptional(true);
    public static final AttributeDescriptor<Integer> EFFICIENCY_EXPORT = GasAsset.EFFICIENCY_EXPORT.withOptional(true);
    public static final AttributeDescriptor<Double> TARIFF_IMPORT = GasSupplierAsset.TARIFF_IMPORT.withOptional(true);
    public static final AttributeDescriptor<Double> TARIFF_EXPORT = GasSupplierAsset.TARIFF_EXPORT.withOptional(true);
    public static final AttributeDescriptor<Double> CARBON_IMPORT = GasSupplierAsset.CARBON_IMPORT.withOptional(true);

    public static final AttributeDescriptor<Double> POWER_FORECAST = new AttributeDescriptor<>("powerForecast", ValueType.NUMBER
    ).withUnits(UNITS_KILO, UNITS_WATT);

    /**
     * For use by hydrators (i.e. JPA/Jackson)
     */
    protected GasConsumerAsset() {
    }

    public GasConsumerAsset(String name) {
        super(name);
    }
}
