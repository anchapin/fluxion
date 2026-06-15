# frozen_string_literal: true

# Replace Constant-Volume Terminal Boxes with VAV Terminals
#
# Finds all AirTerminal:SingleDuct:Uncontrolled (CV) objects in the model,
# removes them, and replaces each with an AirTerminal:SingleDuct:VAV:Reheat
# terminal using an electric reheat coil. A minimum airflow fraction is applied
# to prevent the VAV box from throttling below the specified ratio of peak flow.
#
# ASHRAE 90.1 Section 6.4.3 requires VAV terminal control with minimum
# position settings. A 0.3 minimum fraction satisfies the minimum outdoor air
# requirement at part load while allowing turndown for energy savings.

require 'openstudio'
require 'openstudio/measure/ShowRunnerOutput'

class ReplaceCvWithVav < OpenStudio::Measure::ModelMeasure
  def name
    'Replace Constant-Volume Terminals with VAV'
  end

  def description
    'Replaces all AirTerminal:SingleDuct:Uncontrolled (CV) boxes with ' \
    'AirTerminal:SingleDuct:VAV:Reheat terminals using electric reheat, ' \
    'applying a configurable minimum airflow fraction.'
  end

  def modeler_description
    'Iterates over all AirTerminalSingleDuctUncontrolled objects. For each: ' \
    'extracts the air loop and zone connections, creates an electric reheat coil ' \
    'and a new AirTerminalSingleDuctVAVReheat, sets the minimum airflow fraction, ' \
    'reconnects to the air loop demand inlet node and zone, then removes the old ' \
    'CV terminal. Minimum airflow fraction defaults to 0.3 per ASHRAE 90.1 ' \
    'Section 6.4.3 guidance for maintaining ventilation at part load.'
  end

  def arguments(model)
    args = OpenStudio::Measure::OSArgumentVector.new

    arg = OpenStudio::Measure::OSArgument.makeDoubleArgument('min_airflow_fraction', true)
    arg.setDisplayName('Minimum Airflow Fraction')
    arg.setDescription(
      'Fraction of maximum airflow that the VAV box may not throttle below. ' \
      'Per ASHRAE 90.1 Section 6.4.3, this ensures ventilation is maintained at part load. ' \
      'Typical range: 0.2-0.5.'
    )
    arg.setDefaultValue(0.3)
    args << arg

    arg = OpenStudio::Measure::OSArgument.makeBoolArgument('apply_sizing', true)
    arg.setDisplayName('Apply Sizing Run')
    arg.setDescription('Whether to request a sizing run after modifications to recalculate airflows.')
    arg.setDefaultValue(true)
    args << arg

    args
  end

  def run(model, runner, user_arguments)
    super(model, runner, user_arguments)

    unless runner.validateUserArguments(arguments(model), user_arguments)
      return false
    end

    min_fraction = runner.getDoubleArgumentValue('min_airflow_fraction', user_arguments)
    apply_sizing = runner.getBoolArgumentValue('apply_sizing', user_arguments)

    unless min_fraction > 0.0 && min_fraction <= 1.0
      runner.registerError(
        "Minimum airflow fraction #{min_fraction} is outside physical bounds (0.0, 1.0]."
      )
      return false
    end

    cv_terminals = model.getAirTerminalSingleDuctUncontrolleds
    initial_count = cv_terminals.size

    if initial_count.zero?
      runner.registerAsNotApplicable('No AirTerminal:SingleDuct:Uncontrolled objects found in model.')
      return true
    end

    runner.registerInitialCondition(
      "Model contains #{initial_count} constant-volume terminal box(es)."
    )

    replaced = 0

    cv_terminals.each do |cv|
      cv_name = cv.name.get

      # Determine the air loop and zone this CV terminal serves.
      # CV terminals sit on the demand side of an air loop between the
      # demand inlet node and the zone.
      air_loop = cv.airLoopHVAC
      if air_loop.empty?
        runner.registerWarning(
          "CV terminal '#{cv_name}' is not connected to an air loop. Skipping."
        )
        next
      end
      air_loop = air_loop.get

      # Find the thermal zone served by this terminal.
      # The outlet node of the CV terminal connects to the zone inlet.
      outlet_node = cv.outletModelObject
      if outlet_node.empty?
        runner.registerWarning(
          "CV terminal '#{cv_name}' has no outlet node. Skipping."
        )
        next
      end
      outlet_node = outlet_node.get.to_Node
      if outlet_node.empty?
        runner.registerWarning(
          "CV terminal '#{cv_name}' outlet is not a node. Skipping."
        )
        next
      end
      outlet_node = outlet_node.get

      zone = nil
      model.getThermalZones.each do |tz|
        equip = tz.zoneEquipment
        equip.each do |ze|
          if ze.name.get == cv_name
            zone = tz
            break
          end
        end
        break if zone
      end

      unless zone
        runner.registerWarning(
          "CV terminal '#{cv_name}' is not assigned to any thermal zone. Skipping."
        )
        next
      end

      inlet_node = cv.inletModelObject
      if inlet_node.empty?
        runner.registerWarning(
          "CV terminal '#{cv_name}' has no inlet node. Skipping."
        )
        next
      end
      inlet_node = inlet_node.get.to_Node
      if inlet_node.empty?
        runner.registerWarning(
          "CV terminal '#{cv_name}' inlet is not a node. Skipping."
        )
        next
      end
      inlet_node = inlet_node.get

      # Create an electric reheat coil for the VAV terminal.
      reheat_coil = OpenStudio::Model::CoilHeatingElectric.new(model)
      reheat_coil.setName("#{cv_name} VAV Reheat Coil")
      reheat_coil.setEfficiency(1.0)

      # Create the VAV:Reheat terminal.
      availability_schedule = model.alwaysOnSchedule
      vav = OpenStudio::Model::AirTerminalSingleDuctVAVReheat.new(
        model, availability_schedule, reheat_coil
      )
      vav.setName("#{cv_name} VAV")
      vav.setMinimumAirFlowFraction(min_fraction)

      # Disconnect the old CV terminal from the air loop.
      cv.disconnect

      # Remove the old CV terminal.
      cv.remove

      # Add the new VAV terminal to the air loop demand side.
      air_loop.addBranchForZone(zone, vav)

      runner.registerInfo(
        "Replaced CV terminal '#{cv_name}' with VAV terminal '#{vav.name.get}' " \
        "(min fraction=#{min_fraction}) serving zone '#{zone.name.get}'."
      )
      replaced += 1
    end

    if replaced.zero?
      runner.registerAsNotApplicable('No CV terminals were eligible for replacement.')
      return true
    end

    if apply_sizing
      runner.registerInfo(
        'Sizing run requested. Ensure the OpenStudio workflow executes a ' \
        'sizing run to recalculate zone design airflows with the new VAV terminals.'
      )
    end

    runner.registerFinalCondition(
      "Replaced #{replaced} of #{initial_count} constant-volume terminal box(es) " \
      "with VAV terminals (min airflow fraction=#{min_fraction})."
    )
    true
  end
end

ReplaceCvWithVav.new.registerWithApplication
